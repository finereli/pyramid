import click
import sqlite3
from datetime import datetime, UTC
from db import init_db, get_session, Observation, Model, Summary
from llm import extract_observations, client, MODEL
from summarize import run_tier0_summarization, run_higher_tier_summarization, run_all_summarization
from embeddings import embed_many, enable_vec, init_memory_vec, get_existing_embeddings, store_embeddings, search_memory, enrich_for_embedding
from loaders import load_glenn_messages, load_claude_messages, load_openclaw_messages, group_messages_by_week
from generate import export_models


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


@cli.command('import', help='Import conversations and extract observations.')
@click.option('--source', default=None, help='Path to source file (optional for openclaw)')
@click.option('--glenn', 'format', flag_value='glenn', help='Glenn SQLite format')
@click.option('--claude', 'format', flag_value='claude', help='Claude JSON format')
@click.option('--openclaw', 'format', flag_value='openclaw', help='OpenClaw JSONL sessions')
@click.option('--limit', '-n', default=None, type=int, help='Limit number of messages to process')
@click.option('--conversation', '-c', default=None, type=int, help='Process specific conversation ID (glenn only)')
@click.option('--user', '-u', default=None, type=str, help='Filter by username (glenn only)')
@click.option('--parallel', '-p', default=10, type=int, help='Number of parallel workers (default: 10)')
@click.option('--no-summarize', is_flag=True, help='Skip summarization during import')
@click.option('--clean', is_flag=True, help='Delete all existing data before import')
def import_cmd(source, format, limit, conversation, user, parallel, no_summarize, clean):
    if not format:
        click.echo('Error: Must specify --glenn, --claude, or --openclaw format')
        return
    
    if format != 'openclaw' and not source:
        click.echo('Error: --source is required for glenn and claude formats')
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
        messages, info = load_glenn_messages(source, conversation, user, limit)
        if info:
            click.echo(info)
    elif format == 'claude':
        messages, _ = load_claude_messages(source, limit)
    else:
        messages, _ = load_openclaw_messages(source, limit)
    
    source_desc = source or 'default openclaw sessions'
    click.echo(f'Loaded {len(messages)} messages from {source_desc}')
    
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
@click.option('--parallel', '-p', default=10, help='Number of parallel workers for batch processing')
@click.option('--force', is_flag=True, help='Clear existing embeddings and re-embed everything')
def embed(parallel, force):
    session = get_session()
    conn = sqlite3.connect('memory.db')
    enable_vec(conn)
    init_memory_vec(conn)
    
    if force:
        conn.execute("DELETE FROM memory_vec")
        conn.commit()
        click.echo('Cleared existing embeddings.')
    
    existing = get_existing_embeddings(conn)
    
    to_embed = []
    for obs in session.query(Observation).all():
        if ('observation', obs.id) not in existing and obs.text and obs.text.strip():
            enriched = enrich_for_embedding(obs.text, obs.timestamp)
            to_embed.append(('observation', obs.id, enriched))
    for s in session.query(Summary).all():
        if ('summary', s.id) not in existing and s.text and s.text.strip():
            enriched = enrich_for_embedding(s.text, s.start_timestamp, s.end_timestamp)
            to_embed.append(('summary', s.id, enriched))
    
    if not to_embed:
        click.echo('Nothing to embed.')
        return
    
    click.echo(f'Embedding {len(to_embed)} items...')
    texts = [item[2] for item in to_embed]
    
    def progress(items_done, items_total, batches_done, batches_total):
        click.echo(f'  {items_done}/{items_total} items ({batches_done}/{batches_total} batches)')
    
    embeddings = embed_many(texts, max_workers=parallel, on_progress=progress)
    
    store_embeddings(conn, to_embed, embeddings)
    conn.close()
    session.close()
    click.echo(f'Done. {len(to_embed)} items embedded.')


@cli.command(help='Semantic search across memory.')
@click.argument('query')
@click.option('--limit', '-n', default=20, help='Number of results to retrieve')
@click.option('--raw', is_flag=True, help='Show raw results without LLM synthesis')
@click.option('--time-weight', '-t', default=0.3, help='Time decay weight (0=pure semantic, 1=heavy recency bias)')
def search(query, limit, raw, time_weight):
    session = get_session()
    conn = sqlite3.connect('memory.db')
    enable_vec(conn)
    
    results = search_memory(conn, query, limit, time_weight=time_weight)
    
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


@cli.command(help='Generate markdown files from models.')
@click.argument('workspace')
@click.option('--db', default='memory.db', help='Path to database file')
@click.option('--debug', is_flag=True, help='Include source info (tier, id, date range)')
@click.option('--no-synthesize', is_flag=True, help='Skip LLM synthesis, just concatenate summaries')
@click.option('--parallel', '-p', default=10, type=int, help='Number of parallel workers (default: 10)')
@click.option('--ref-date', default=None, help='Reference date for time buckets (YYYY-MM-DD, default: today)')
def generate(workspace, db, debug, no_synthesize, parallel, ref_date):
    from datetime import datetime, UTC
    
    parsed_ref_date = None
    if ref_date:
        parsed_ref_date = datetime.strptime(ref_date, '%Y-%m-%d').replace(tzinfo=UTC)
    
    progress = lambda msg: click.echo(msg)
    export_models(workspace, db, debug, do_synthesize=not no_synthesize, on_progress=progress, max_workers=parallel, ref_date=parsed_ref_date)
    click.echo('Done')


if __name__ == '__main__':
    cli()
