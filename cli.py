import click
import sqlite3
from datetime import datetime, UTC
from sqlalchemy import create_engine, text
from db import init_db, get_session, get_engine, Observation, Model, Summary
from llm import extract_observations
from summarize import run_tier0_summarization, run_higher_tier_summarization, run_all_summarization
from embeddings import get_embedding, enable_vec, serialize_embedding, EMBEDDING_DIM


@click.group()
def cli():
    init_db()


@cli.command()
@click.argument('text')
@click.option('--importance', '-i', default=5, help='Importance 1-10+')
def observe(text, importance):
    session = get_session()
    obs = Observation(text=text, importance=importance, timestamp=datetime.now(UTC))
    session.add(obs)
    session.commit()
    click.echo(f'Added observation #{obs.id} (importance={importance})')
    session.close()


@cli.command('list')
@click.option('--limit', '-n', default=20, help='Number of observations to show')
def list_observations(limit):
    session = get_session()
    observations = session.query(Observation).order_by(Observation.timestamp.desc()).limit(limit).all()
    
    if not observations:
        click.echo('No observations yet.')
        return
    
    for obs in reversed(observations):
        model_name = obs.model.name if obs.model else '-'
        click.echo(f'[{obs.id}] {obs.timestamp:%Y-%m-%d %H:%M} ({obs.importance}) [{model_name}] {obs.text}')
    
    session.close()


@cli.command()
def models():
    session = get_session()
    all_models = session.query(Model).all()
    
    for m in all_models:
        base_marker = '*' if m.is_base else ' '
        desc = m.description or '(no description)'
        click.echo(f'{base_marker} {m.name}: {desc}')
    
    session.close()


@cli.command()
@click.option('--source', required=True, help='Path to source database')
@click.option('--limit', '-n', default=None, type=int, help='Limit number of messages to process')
@click.option('--conversation', '-c', default=None, type=int, help='Process specific conversation ID')
def bootstrap(source, limit, conversation):
    source_engine = create_engine(f'sqlite:///{source}')
    
    query = "SELECT role, content, timestamp FROM messages WHERE content IS NOT NULL AND content != ''"
    if conversation:
        query += f" AND conversation_id = {conversation}"
    query += " ORDER BY timestamp"
    if limit:
        query += f" LIMIT {limit}"
    
    with source_engine.connect() as conn:
        result = conn.execute(text(query))
        messages = [{'role': row[0], 'content': row[1], 'timestamp': row[2]} for row in result]
    
    click.echo(f'Loaded {len(messages)} messages from {source}')
    
    if not messages:
        click.echo('No messages to process.')
        return
    
    def progress(chunk_num, total_chunks, msgs_in_chunk, timestamp, obs_count=None):
        ts_str = timestamp[:10] if timestamp else '?'
        if obs_count is None:
            click.echo(f'  Chunk {chunk_num}/{total_chunks} ({msgs_in_chunk} msgs, {ts_str})...', nl=False)
        else:
            click.echo(f' {obs_count} obs')
    
    click.echo('Extracting observations...')
    observations = extract_observations(messages, on_progress=progress)
    click.echo(f'Total: {len(observations)} observations')
    
    session = get_session()
    for obs_data in observations:
        ts = obs_data.get('timestamp')
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        obs = Observation(
            text=obs_data['text'],
            importance=obs_data['importance'],
            timestamp=ts or datetime.now(UTC)
        )
        session.add(obs)
    session.commit()
    session.close()
    
    click.echo(f'Saved {len(observations)} observations to memory.db')


@cli.command()
@click.option('--tier', '-t', default=None, type=int, help='Run only specific tier (0 for observations, omit for all)')
def summarize(tier):
    progress = lambda msg: click.echo(f'  {msg}')
    
    if tier == 0:
        click.echo('Running tier 0 summarization...')
        created = run_tier0_summarization(on_progress=progress)
        click.echo(f'Created {created} tier 0 summaries')
    elif tier is not None:
        click.echo(f'Running tier {tier} summarization...')
        created = run_higher_tier_summarization(on_progress=progress)
        click.echo(f'Created {created} higher tier summaries')
    else:
        click.echo('Running full summarization...')
        tier0, higher = run_all_summarization(on_progress=progress)
        click.echo(f'Created {tier0} tier 0 + {higher} higher tier summaries')


@cli.command()
@click.option('--tier', '-t', default=None, type=int, help='Filter by tier')
def summaries(tier):
    session = get_session()
    query = session.query(Summary).order_by(Summary.tier, Summary.start_timestamp)
    if tier is not None:
        query = query.filter(Summary.tier == tier)
    
    for s in query.all():
        model_name = s.model.name if s.model else '-'
        click.echo(f'[T{s.tier}] {s.start_timestamp:%Y-%m-%d} - {s.end_timestamp:%Y-%m-%d} [{model_name}]')
        click.echo(f'  {s.text}')
        click.echo()
    
    session.close()


@cli.command()
@click.argument('name')
def model(name):
    session = get_session()
    m = session.query(Model).filter_by(name=name).first()
    
    if not m:
        click.echo(f'Model "{name}" not found.')
        session.close()
        return
    
    click.echo(f'Model: {m.name}')
    click.echo(f'Description: {m.description or "(none)"}')
    click.echo()
    
    summaries = session.query(Summary).filter_by(model_id=m.id).order_by(Summary.tier.desc(), Summary.start_timestamp.desc()).all()
    
    current_tier = None
    for s in summaries:
        if s.tier != current_tier:
            current_tier = s.tier
            click.echo(f'--- Tier {s.tier} ---')
        click.echo(f'{s.start_timestamp:%Y-%m-%d} - {s.end_timestamp:%Y-%m-%d}')
        click.echo(f'  {s.text}')
        click.echo()
    
    session.close()


@cli.command()
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


@cli.command()
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


if __name__ == '__main__':
    cli()
