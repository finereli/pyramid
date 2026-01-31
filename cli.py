import click
from datetime import datetime, UTC
from sqlalchemy import create_engine, text
from db import init_db, get_session, Observation, Model, Summary
from llm import extract_observations
from summarize import run_tier0_summarization


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
def summarize():
    click.echo('Running tier 0 summarization...')
    created = run_tier0_summarization(on_progress=lambda msg: click.echo(f'  {msg}'))
    click.echo(f'Created {created} tier 0 summaries')


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


if __name__ == '__main__':
    cli()
