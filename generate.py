from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import func
from db import get_session, Model, Observation
from llm import client, MODEL


CORE_MODELS = ['assistant', 'user']


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


def render_memory(assistant_content, user_content, other_models):
    lines = [
        '# Memory',
        '',
        'Synthesized memory from conversations. SOUL.md and USER.md are identity files and not overwritten.',
        '',
        '---',
        '',
        '## Self',
        '',
        assistant_content or '*No assistant observations yet.*',
        '',
        '---',
        '',
        '## User',
        '',
        user_content or '*No user observations yet.*',
        '',
    ]
    
    if other_models:
        lines.append('---')
        lines.append('')
        lines.append('## Other Models')
        lines.append('')
        for model, path in other_models:
            desc = model.description or ''
            lines.append(f'- [{path}]({path}): {desc}')
        lines.append('')
    
    return '\n'.join(lines)


def write_file(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() == content:
        return False
    path.write_text(content)
    return True


def render_model_file(model):
    lines = [
        '---',
        f'name: {model.name}',
        f'description: {model.description or ""}',
        '---',
        '',
        f'# {model.name.title()}',
        '',
        model.synthesized_content or '*No content yet.*',
        '',
    ]
    return '\n'.join(lines)


def export_models(workspace, db_path='pyramid.db', on_progress=None, max_workers=10, model_ids=None):
    workspace = Path(workspace)
    session = get_session(db_path)
    
    update_model_descriptions(session, on_progress, max_workers)
    
    if model_ids is not None:
        models = session.query(Model).filter(Model.id.in_(model_ids)).all()
    else:
        models = session.query(Model).all()
    
    regenerated = []
    
    assistant_model = None
    user_model = None
    other_models = []
    
    for model in models:
        if model.name == 'assistant':
            assistant_model = model
        elif model.name == 'user':
            user_model = model
        else:
            other_models.append(model)
    
    for model in other_models:
        content = render_model_file(model)
        path = workspace / f'models/{model.name}.md'
        if write_file(path, content):
            regenerated.append(f'models/{model.name}.md')
    
    write_memory = model_ids is None or any(m.name in CORE_MODELS for m in models)
    if write_memory:
        all_other_models = session.query(Model).filter(~Model.name.in_(CORE_MODELS)).all()
        other_models_index = [
            (m, f'models/{m.name}.md')
            for m in all_other_models
        ]
        assistant_content = assistant_model.synthesized_content if assistant_model else None
        user_content = user_model.synthesized_content if user_model else None
        memory_content = render_memory(assistant_content, user_content, other_models_index)
        if write_file(workspace / 'MEMORY.md', memory_content):
            regenerated.append('MEMORY.md')
    
    session.close()
    
    return regenerated
