import click
import sqlite3
import json
import hashlib
from pathlib import Path
from datetime import datetime, UTC
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import create_engine, text, func
from db import init_db, get_session, get_engine, Observation, Model, Summary
from llm import extract_observations, client, MODEL
from summarize import run_tier0_summarization, run_higher_tier_summarization, run_all_summarization
from embeddings import get_embedding, enable_vec, serialize_embedding, EMBEDDING_DIM
from pyramid import get_pyramid, get_unsummarized_observations, synthesize_model


@click.group()
def cli():
    init_db()


@cli.command(help='Add a single observation manually.')
@click.argument('text')
def observe(text):
    session = get_session()
    obs = Observation(text=text, timestamp=datetime.now(UTC))
    session.add(obs)
    session.commit()
    click.echo(f'Added observation #{obs.id}')
    session.close()


def get_week_key(timestamp_str):
    if not timestamp_str:
        return 'unknown'
    dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    year, week, _ = dt.isocalendar()
    return f'{year}-W{week:02d}'


def group_messages_by_week(messages):
    by_week = {}
    for msg in messages:
        week = get_week_key(msg.get('timestamp', ''))
        if week not in by_week:
            by_week[week] = []
        by_week[week].append(msg)
    return by_week


def load_glenn_messages(source, conversation, user, limit):
    source_engine = create_engine(f'sqlite:///{source}')
    
    user_id = None
    if user:
        with source_engine.connect() as conn:
            result = conn.execute(text("SELECT id FROM users WHERE LOWER(username) = :username"), {'username': user.lower()})
            row = result.fetchone()
            if not row:
                click.echo(f"User '{user}' not found in source database")
                return []
            user_id = row[0]
            click.echo(f"Filtering for user '{user}' (id={user_id})")
    
    if user_id:
        query = """SELECT m.role, m.content, m.timestamp 
                   FROM messages m 
                   JOIN conversations c ON m.conversation_id = c.id 
                   WHERE m.content IS NOT NULL AND m.content != '' AND c.user_id = :user_id"""
    else:
        query = "SELECT role, content, timestamp FROM messages WHERE content IS NOT NULL AND content != ''"
    
    if conversation:
        query += f" AND conversation_id = {conversation}"
    query += " ORDER BY timestamp"
    if limit:
        query += f" LIMIT {limit}"
    
    with source_engine.connect() as conn:
        if user_id:
            result = conn.execute(text(query), {'user_id': user_id})
        else:
            result = conn.execute(text(query))
        return [{'role': row[0], 'content': row[1], 'timestamp': row[2]} for row in result]


def load_claude_messages(source, limit):
    with open(source) as f:
        data = json.load(f)
    
    messages = []
    for conv in data:
        for msg in conv.get('chat_messages', []):
            sender = msg.get('sender', '')
            role = 'user' if sender == 'human' else 'assistant'
            timestamp = msg.get('created_at', '')
            
            content_parts = []
            for content in msg.get('content', []):
                if content.get('type') == 'text' and content.get('text'):
                    content_parts.append(content['text'])
            
            if content_parts:
                messages.append({
                    'role': role,
                    'content': '\n'.join(content_parts),
                    'timestamp': timestamp
                })
    
    messages.sort(key=lambda m: m.get('timestamp', ''))
    if limit:
        messages = messages[:limit]
    return messages


@cli.command('import', help='Import conversations and extract observations.')
@click.option('--source', required=True, help='Path to source file')
@click.option('--glenn', 'format', flag_value='glenn', help='Glenn SQLite format')
@click.option('--claude', 'format', flag_value='claude', help='Claude JSON format')
@click.option('--limit', '-n', default=None, type=int, help='Limit number of messages to process')
@click.option('--conversation', '-c', default=None, type=int, help='Process specific conversation ID (glenn only)')
@click.option('--user', '-u', default=None, type=str, help='Filter by username (glenn only)')
@click.option('--parallel', '-p', default=10, type=int, help='Number of parallel workers (default: 10)')
@click.option('--no-summarize', is_flag=True, help='Skip summarization during import')
@click.option('--clean', is_flag=True, help='Delete all existing data before import')
def import_cmd(source, format, limit, conversation, user, parallel, no_summarize, clean):
    if not format:
        click.echo('Error: Must specify --glenn or --claude format')
        return
    
    if clean:
        session = get_session()
        deleted_obs = session.query(Observation).delete()
        deleted_summaries = session.query(Summary).delete()
        deleted_models = session.query(Model).filter(Model.is_base == False).delete()
        session.commit()
        session.close()
        click.echo(f'Cleaned: {deleted_obs} observations, {deleted_summaries} summaries, {deleted_models} models')
    
    if format == 'glenn':
        messages = load_glenn_messages(source, conversation, user, limit)
    else:
        messages = load_claude_messages(source, limit)
    
    click.echo(f'Loaded {len(messages)} messages from {source}')
    
    if not messages:
        click.echo('No messages to process.')
        return
    
    by_week = group_messages_by_week(messages)
    weeks = sorted(by_week.keys())
    click.echo(f'Processing {len(weeks)} weeks...')
    
    total_observations = 0
    
    for week in weeks:
        week_messages = by_week[week]
        click.echo(f'\n{week}: {len(week_messages)} messages')
        
        def progress(completed, total, msgs_in_chunk, timestamp, obs_count):
            click.echo(f'  [{completed}/{total}] {obs_count} obs')
        
        observations = extract_observations(week_messages, on_progress=progress, max_workers=parallel)
        
        session = get_session()
        for obs_data in observations:
            ts = obs_data.get('timestamp')
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            obs = Observation(
                text=obs_data['text'],
                timestamp=ts or datetime.now(UTC)
            )
            session.add(obs)
        session.commit()
        session.close()
        
        total_observations += len(observations)
        click.echo(f'  Saved {len(observations)} observations')
        
        if not no_summarize:
            progress = lambda msg: click.echo(f'    {msg}')
            created = run_tier0_summarization(on_progress=progress)
            if created:
                click.echo(f'  Created {created} tier 0 summaries')
    
    click.echo(f'\nTotal: {total_observations} observations')
    
    if not no_summarize:
        click.echo('\nRunning higher tier summarization...')
        progress = lambda msg: click.echo(f'  {msg}')
        higher = run_higher_tier_summarization(on_progress=progress)
        click.echo(f'Created {higher} higher tier summaries')


@cli.command(help='Run summarization to compress observations.')
@click.option('--start', '-s', default=None, type=int, help='Start from observation ID (skip earlier)')
@click.option('--max-obs', '-n', default=None, type=int, help='Maximum observations to process (for testing)')
@click.option('--max-tier', '-T', default=None, type=int, help='Maximum tier to build (e.g., 1 = only tier 0 and 1)')
@click.option('--parallel', '-p', default=10, type=int, help='Number of parallel workers')
@click.option('--clean', is_flag=True, help='Delete all existing summaries and model assignments before running')
def summarize(start, max_obs, max_tier, parallel, clean):
    session = get_session()
    
    if clean:
        deleted_summaries = session.query(Summary).delete()
        session.query(Observation).update({Observation.model_id: None})
        deleted_models = session.query(Model).filter(Model.is_base == False).delete()
        session.commit()
        click.echo(f'Cleaned: {deleted_summaries} summaries, {deleted_models} non-base models, reset assignments')
    
    session.close()
    
    progress = lambda msg: click.echo(msg)
    
    click.echo('Running summarization...')
    tier0, higher = run_all_summarization(on_progress=progress, max_workers=parallel, max_tier=max_tier, max_obs=max_obs, start_id=start)
    click.echo(f'Created {tier0} tier 0 + {higher} higher tier summaries')


@cli.command(help='Generate embeddings for semantic search.')
def embed():
    session = get_session()
    
    conn = sqlite3.connect('memory.db')
    enable_vec(conn)
    
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0(
            id INTEGER PRIMARY KEY,
            source_type TEXT,
            source_id INTEGER,
            embedding float[{EMBEDDING_DIM}]
        )
    """)
    
    existing = set(
        (row[0], row[1]) 
        for row in conn.execute("SELECT source_type, source_id FROM memory_vec").fetchall()
    )
    
    observations = session.query(Observation).all()
    summaries = session.query(Summary).all()
    
    to_embed = []
    for obs in observations:
        if ('observation', obs.id) not in existing:
            to_embed.append(('observation', obs.id, obs.text))
    for s in summaries:
        if ('summary', s.id) not in existing:
            to_embed.append(('summary', s.id, s.text))
    
    click.echo(f'Embedding {len(to_embed)} items...')
    
    for i, (source_type, source_id, text) in enumerate(to_embed):
        embedding = get_embedding(text)
        conn.execute(
            "INSERT INTO memory_vec (source_type, source_id, embedding) VALUES (?, ?, ?)",
            [source_type, source_id, serialize_embedding(embedding)]
        )
        if (i + 1) % 10 == 0:
            click.echo(f'  {i + 1}/{len(to_embed)}')
            conn.commit()
    
    conn.commit()
    conn.close()
    session.close()
    click.echo(f'Done. {len(to_embed)} items embedded.')


@cli.command(help='Semantic search across memory.')
@click.argument('query')
@click.option('--limit', '-n', default=20, help='Number of results to retrieve')
@click.option('--raw', is_flag=True, help='Show raw results without LLM synthesis')
def search(query, limit, raw):
    from llm import client, MODEL
    
    session = get_session()
    
    conn = sqlite3.connect('memory.db')
    enable_vec(conn)
    
    query_embedding = get_embedding(query)
    
    results = conn.execute(f"""
        SELECT source_type, source_id, distance
        FROM memory_vec
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
    """, [serialize_embedding(query_embedding), limit]).fetchall()
    
    if not results:
        click.echo('No results. Run "embed" first to create embeddings.')
        return
    
    context_items = []
    for source_type, source_id, distance in results:
        if source_type == 'observation':
            item = session.get(Observation, source_id)
            if item:
                context_items.append(f"[obs] {item.text}")
        else:
            item = session.get(Summary, source_id)
            if item:
                model_name = item.model.name if item.model else '?'
                context_items.append(f"[{model_name} T{item.tier}] {item.text}")
    
    conn.close()
    
    if raw:
        for i, (source_type, source_id, distance) in enumerate(results):
            click.echo(f'[{distance:.3f}] {context_items[i]}')
        session.close()
        return
    
    context = "\n\n".join(context_items)
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Answer questions based on the memory context provided. Be concise and direct. If the answer isn't in the context, say so."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )
    
    click.echo(response.choices[0].message.content)
    session.close()


CORE_MODEL_FILES = {
    'assistant': 'SOUL.md',
    'user': 'USER.md',
    'system': 'TOOLS.md',
}

TIER_LABELS = {
    0: 'Recent',
    1: 'This Month',
    2: 'Historical',
}


def update_model_descriptions(session, on_progress=None):
    models = session.query(Model).filter(
        (Model.description == None) | (Model.description == '')
    ).filter(Model.is_base == False).all()
    
    if not models:
        return
    
    if on_progress:
        on_progress(f"Deriving descriptions for {len(models)} models...")
    
    def derive_one(model_id, model_name):
        samples = session.query(Observation).filter(
            Observation.model_id == model_id
        ).order_by(func.random()).limit(10).all()
        
        if not samples:
            return model_id, None
        
        sample_text = "\n".join(f"- {s.text}" for s in samples)
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user", 
                "content": f"""These observations are stored under the model '{model_name}':

{sample_text}

Write a brief description (under 120 chars) with format: "[Who/What this is] - [what kind of info is stored]"
Examples:
- "Tony Ennis (coaching client) - business challenges, relationship dynamics, session notes"
- "GrowthLab Consulting (business) - financials, pricing, client segments, service offerings"
- "Corfu Travel (trip) - logistics, itinerary, health precautions, cultural notes\""""
            }]
        )
        desc = response.choices[0].message.content.strip()
        if len(desc) > 120:
            desc = desc[:117] + '...'
        return model_id, desc
    
    results = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(derive_one, m.id, m.name): m for m in models}
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
    workspace = Path(workspace)
    session = get_session(db_path)
    cache = load_cache(workspace)
    
    update_model_descriptions(session, on_progress)
    
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
    
    def render_one(data):
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


@cli.command(help='Generate markdown files from models.')
@click.argument('workspace')
@click.option('--db', default='memory.db', help='Path to database file')
@click.option('--force', is_flag=True, help='Force regenerate all files')
@click.option('--debug', is_flag=True, help='Include source info (tier, id, date range)')
@click.option('--no-synthesize', is_flag=True, help='Skip LLM synthesis, just concatenate summaries')
def generate(workspace, db, force, debug, no_synthesize):
    progress = lambda msg: click.echo(msg)
    changed = export_models(workspace, db, force, debug, do_synthesize=not no_synthesize, on_progress=progress)
    click.echo(f'Updated {changed} files')


if __name__ == '__main__':
    cli()
