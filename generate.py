from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import func
from db import get_session, Model, Observation
from llm import client, MODEL
from pyramid import get_pyramid, get_unsummarized_observations, synthesize_model


CORE_MODEL_FILES = {
    'assistant': 'SOUL.md',
    'user': 'USER.md',
}

TIER_LABELS = {
    0: 'Recent',
    1: 'This Month',
    2: 'Historical',
}


def update_model_descriptions(session, on_progress=None, max_workers=10):
    models = session.query(Model).filter(
        (Model.description == None) | (Model.description == '')
    ).filter(Model.is_base == False).all()
    
    if not models:
        return
    
    if on_progress:
        on_progress(f"Deriving descriptions for {len(models)} models...")
    
    model_samples = {}
    for model in models:
        samples = session.query(Observation).filter(
            Observation.model_id == model.id
        ).order_by(func.random()).limit(10).all()
        model_samples[model.id] = [s.text for s in samples]
    
    def derive_one(model_id, model_name, samples):
        if not samples:
            return model_id, None
        
        sample_text = "\n".join(f"- {s}" for s in samples)
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user", 
                "content": f"""These observations are stored under the model '{model_name}':

{sample_text}

Write a brief description (under 120 chars) with format: "[Who/What this is] - [what kind of info is stored]"
Examples:
- "Marcus Chen (mentee) - career goals, skill development, meeting notes"
- "Sunrise Bakery (business) - recipes, suppliers, seasonal menu planning"
- "Japan 2025 (trip) - flights, accommodations, restaurant reservations, packing list\""""
            }]
        )
        desc = response.choices[0].message.content.strip()
        if len(desc) > 120:
            desc = desc[:117] + '...'
        return model_id, desc
    
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(derive_one, m.id, m.name, model_samples[m.id]): m for m in models}
        for i, future in enumerate(as_completed(futures), 1):
            model = futures[future]
            model_id, desc = future.result()
            results[model_id] = desc
            if on_progress:
                on_progress(f"  [{i}/{len(models)}] {model.name}")
    
    for model in models:
        if results.get(model.id):
            model.description = results[model.id]
    
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


def write_file(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def render_model(data, do_synthesize=True, debug=False):
    by_tier = data['by_tier']
    unsummarized = data['unsummarized']
    unsummarized_ts = data.get('unsummarized_ts', [])
    ref_date = data.get('ref_date')
    
    unsummarized_with_ts = list(zip(unsummarized, unsummarized_ts)) if unsummarized_ts else [(t, ref_date) for t in unsummarized]
    
    if do_synthesize and (by_tier or unsummarized) and not debug:
        synthesized = synthesize_model(data['name'], data['description'], by_tier, unsummarized_with_ts, ref_date)
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
                lines.append(f'### T{tier} ({s["end_timestamp"]:%Y-%m-%d})')
                lines.append('')
            lines.append(s['text'])
            lines.append('')
    
    if unsummarized:
        lines.append('## Unsummarized')
        lines.append('')
        for text in unsummarized:
            lines.append(f'- {text}')
        lines.append('')
    
    return '\n'.join(lines)


def export_models(workspace, db_path='memory.db', debug=False, do_synthesize=True, on_progress=None, max_workers=10):
    workspace = Path(workspace)
    session = get_session(db_path)
    
    update_model_descriptions(session, on_progress, max_workers)
    
    models = session.query(Model).all()
    
    global_ref_date = session.query(func.max(Observation.timestamp)).scalar()
    
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
            'by_tier': {tier: [{'text': s.text, 'end_timestamp': s.end_timestamp} for s in sums] 
                        for tier, sums in by_tier.items()},
            'unsummarized': [o.text for o in unsummarized],
            'unsummarized_ts': [o.timestamp for o in unsummarized],
            'filename': filename,
            'ref_date': global_ref_date,
        })
    
    session.close()
    
    core_models = []
    other_models = []
    results = {}
    
    if do_synthesize and not debug:
        if on_progress:
            on_progress(f"Synthesizing {len(model_data)} models...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(render_model, data, do_synthesize, debug): data for data in model_data}
            for i, future in enumerate(as_completed(futures), 1):
                data = futures[future]
                results[data['name']] = future.result()
                if on_progress:
                    on_progress(f"  [{i}/{len(model_data)}] {data['name']}")
    else:
        for data in model_data:
            results[data['name']] = render_model(data, do_synthesize, debug)
    
    for data in model_data:
        content = results[data['name']]
        path = workspace / data['filename']
        
        if data['name'] in CORE_MODEL_FILES:
            core_models.append((data, data['filename']))
        else:
            other_models.append((data, data['filename']))
        
        write_file(path, content)
    
    memory_content = render_memory_index(core_models, other_models)
    write_file(workspace / 'MEMORY.md', memory_content)
