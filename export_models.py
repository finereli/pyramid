#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from pathlib import Path
from db import get_session, Model, Summary
from pyramid import get_pyramid

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


def derive_model_description(text):
    lines = text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^\[\d+\]\s*', '', line)
        line = re.sub(r'^(IMPORTANT|CRITICAL|ESSENTIAL):\s*', '', line)
        first_sentence = re.split(r'[.!?]', line)[0].strip()
        if len(first_sentence) > 10:
            if len(first_sentence) > 100:
                first_sentence = first_sentence[:97] + '...'
            return first_sentence
    return None


def update_model_descriptions(session):
    models = session.query(Model).filter(Model.description == None).all()
    models += session.query(Model).filter(Model.description == '').all()
    
    for model in models:
        highest_tier = session.query(Summary).filter(
            Summary.model_id == model.id
        ).order_by(Summary.tier.desc()).first()
        
        if highest_tier:
            description = derive_model_description(highest_tier.text)
            if description:
                model.description = description
    
    session.commit()


def render_model_markdown(session, model):
    lines = [
        '---',
        f'name: {model.name}',
        f'description: {model.description or ""}',
        '---',
        '',
        f'# {model.name.title()}',
        '',
    ]
    
    by_tier = get_pyramid(session, model.id)
    
    for tier in sorted(by_tier.keys()):
        label = TIER_LABELS.get(tier, f'Tier {tier}')
        lines.append(f'## {label}')
        lines.append('')
        for s in by_tier[tier]:
            lines.append(s.text)
            lines.append('')
    
    return '\n'.join(lines)


def render_memory_index(core_models, other_models):
    lines = [
        '# Memory',
        '',
        '## Core',
        '',
    ]
    
    for model, path in core_models:
        lines.append(f'- [{path}]({path}): {model.description or ""}')
    
    if other_models:
        lines.append('')
        lines.append('## Models')
        lines.append('')
        for model, path in other_models:
            lines.append(f'- [{path}]({path}): {model.description or ""}')
    
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


def export_models(workspace, db_path='memory.db', force=False):
    workspace = Path(workspace)
    session = get_session(db_path)
    cache = load_cache(workspace)
    
    update_model_descriptions(session)
    
    models = session.query(Model).all()
    
    core_models = []
    other_models = []
    changed = 0
    
    for model in models:
        content = render_model_markdown(session, model)
        
        if model.name in CORE_MODEL_FILES:
            filename = CORE_MODEL_FILES[model.name]
            path = workspace / filename
            core_models.append((model, filename))
        else:
            filename = f'models/{model.name}.md'
            path = workspace / filename
            other_models.append((model, filename))
        
        if write_if_changed(path, content, cache, force):
            changed += 1
    
    memory_content = render_memory_index(core_models, other_models)
    if write_if_changed(workspace / 'MEMORY.md', memory_content, cache, force):
        changed += 1
    
    save_cache(workspace, cache)
    session.close()
    
    return changed


def main():
    parser = argparse.ArgumentParser(description='Export memory models to markdown files')
    parser.add_argument('workspace', help='Path to workspace directory')
    parser.add_argument('--db', default='memory.db', help='Path to database file')
    parser.add_argument('--force', action='store_true', help='Force regenerate all files')
    args = parser.parse_args()
    
    changed = export_models(args.workspace, args.db, args.force)
    print(f'Updated {changed} files')


if __name__ == '__main__':
    main()
