#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
from db import get_session, Model, Observation
from pyramid import get_pyramid, get_unsummarized_observations, synthesize_model
from llm import client, MODEL
from sqlalchemy import func

CORE_MODEL_FILES = {
    'self': 'SOUL.md',
    'user': 'USER.md',
    'system': 'TOOLS.md',
}

TIER_LABELS = {
    0: 'Recent',
    1: 'This Month',
    2: 'Historical',
}


def derive_model_purpose(session, model):
    samples = session.query(Observation).filter(
        Observation.model_id == model.id
    ).order_by(func.random()).limit(10).all()
    
    if not samples:
        return None
    
    sample_text = "\n".join(f"- {s.text}" for s in samples)
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user", 
            "content": f"""These observations are stored under the model '{model.name}':

{sample_text}

Write a 1-sentence description of what KIND of information belongs in this model.
Focus on category/type, not specific content. Keep it under 100 characters."""
        }]
    )
    desc = response.choices[0].message.content.strip()
    if len(desc) > 100:
        desc = desc[:97] + '...'
    return desc


def update_model_descriptions(session):
    models = session.query(Model).filter(
        (Model.description == None) | (Model.description == '')
    ).filter(Model.is_base == False).all()
    
    for model in models:
        description = derive_model_purpose(session, model)
        if description:
            model.description = description
    
    session.commit()


def render_memory_index(core_models, other_models):
    lines = [
        '# Memory',
        '',
        '## Core',
        '',
    ]
    
    for data, path in core_models:
        desc = data['description'] if isinstance(data, dict) else data.description
        lines.append(f'- [{path}]({path}): {desc or ""}')
    
    if other_models:
        lines.append('')
        lines.append('## Models')
        lines.append('')
        for data, path in other_models:
            desc = data['description'] if isinstance(data, dict) else data.description
            lines.append(f'- [{path}]({path}): {desc or ""}')
    
    lines.append('')
    return '\n'.join(lines)


def content_hash(content):
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def load_cache(workspace):
    cache_file = workspace / '.memory_cache.json'
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    return {}


def save_cache(workspace, cache):
    cache_file = workspace / '.memory_cache.json'
    cache_file.write_text(json.dumps(cache))


def write_if_changed(path, content, cache, force=False):
    h = content_hash(content)
    cache_key = str(path)
    
    if not force and cache.get(cache_key) == h and path.exists():
        return False
    
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    cache[cache_key] = h
    return True


def export_models(workspace, db_path='memory.db', force=False, debug=False, do_synthesize=True, on_progress=None):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    workspace = Path(workspace)
    session = get_session(db_path)
    cache = load_cache(workspace)
    
    update_model_descriptions(session)
    
    models = session.query(Model).all()
    
    model_data = []
    for model in models:
        by_tier = get_pyramid(session, model.id)
        unsummarized = get_unsummarized_observations(session, model.id, by_tier)
        
        if model.name in CORE_MODEL_FILES:
            filename = CORE_MODEL_FILES[model.name]
        else:
            filename = f'models/{model.name}.md'
        
        model_data.append({
            'name': model.name,
            'description': model.description,
            'by_tier': by_tier,
            'unsummarized': unsummarized,
            'filename': filename,
        })
    
    session.close()
    
    def render_one(data):
        by_tier = data['by_tier']
        unsummarized = data['unsummarized']
        
        if do_synthesize and (by_tier or unsummarized) and not debug:
            synthesized = synthesize_model(data['name'], data['description'], by_tier, unsummarized)
            if synthesized:
                lines = [
                    '---',
                    f'name: {data["name"]}',
                    f'description: {data["description"] or ""}',
                    '---',
                    '',
                    f'# {data["name"].title()}',
                    '',
                    synthesized,
                    '',
                ]
                return '\n'.join(lines)
        
        lines = [
            '---',
            f'name: {data["name"]}',
            f'description: {data["description"] or ""}',
            '---',
            '',
            f'# {data["name"].title()}',
            '',
        ]
        
        for tier in sorted(by_tier.keys()):
            label = TIER_LABELS.get(tier, f'Tier {tier}')
            lines.append(f'## {label}')
            lines.append('')
            for s in by_tier[tier]:
                if debug:
                    lines.append(f'### T{tier} #{s.id} ({s.start_timestamp:%Y-%m-%d} to {s.end_timestamp:%Y-%m-%d})')
                    lines.append('')
                lines.append(s.text)
                lines.append('')
        
        if unsummarized:
            lines.append('## Unsummarized')
            lines.append('')
            for o in unsummarized:
                lines.append(f'- {o.text}')
            lines.append('')
        
        return '\n'.join(lines)
    
    core_models = []
    other_models = []
    results = {}
    
    if do_synthesize and not debug:
        if on_progress:
            on_progress(f"Synthesizing {len(model_data)} models...")
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(render_one, data): data for data in model_data}
            for i, future in enumerate(as_completed(futures), 1):
                data = futures[future]
                results[data['name']] = future.result()
                if on_progress:
                    on_progress(f"  [{i}/{len(model_data)}] {data['name']}")
    else:
        for data in model_data:
            results[data['name']] = render_one(data)
    
    changed = 0
    for data in model_data:
        content = results[data['name']]
        path = workspace / data['filename']
        
        if data['name'] in CORE_MODEL_FILES:
            core_models.append((data, data['filename']))
        else:
            other_models.append((data, data['filename']))
        
        if write_if_changed(path, content, cache, force):
            changed += 1
    
    memory_content = render_memory_index(core_models, other_models)
    if write_if_changed(workspace / 'MEMORY.md', memory_content, cache, force):
        changed += 1
    
    save_cache(workspace, cache)
    
    return changed


def main():
    parser = argparse.ArgumentParser(description='Export memory models to markdown files')
    parser.add_argument('workspace', help='Path to workspace directory')
    parser.add_argument('--db', default='memory.db', help='Path to database file')
    parser.add_argument('--force', action='store_true', help='Force regenerate all files')
    parser.add_argument('--debug', action='store_true', help='Include source info (tier, id, date range, observations)')
    parser.add_argument('--no-synthesize', action='store_true', help='Skip LLM synthesis, just concatenate summaries')
    args = parser.parse_args()
    
    progress = lambda msg: print(msg)
    changed = export_models(args.workspace, args.db, args.force, args.debug, do_synthesize=not args.no_synthesize, on_progress=progress)
    print(f'Updated {changed} files')


if __name__ == '__main__':
    main()
