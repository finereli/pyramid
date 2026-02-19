import click
import sqlite3
from pathlib import Path
from datetime import datetime, UTC
from db import init_db, get_session, Observation, Model, Summary, ImportedSession
from llm import extract_observations, client, MODEL
from summarize import run_tier0_summarization, run_higher_tier_summarization, run_all_summarization, mark_model_dirty
from embeddings import embed_many, enable_vec, init_memory_vec, get_existing_embeddings, store_embeddings, search_memory, enrich_for_embedding
from loaders import load_glenn_messages, load_claude_messages, load_openclaw_messages, group_messages_by_week, get_openclaw_file_stats, load_git_commits, load_git_incremental
from llm import GIT_OBSERVE_SYSTEM_PROMPT
from generate import export_models
from pyramid import synthesize_dirty_models
from sync import sync as do_sync


def get_db_path(workspace, db):
    return Path(workspace) / db


@click.group()
def cli():
    pass


@cli.command('import', help='Import conversations and extract observations.')
@click.option('--workspace', '-w', required=True, help='Workspace directory')
@click.option('--db', default='pyramid.db', help='Database filename (default: pyramid.db)')
@click.option('--source', default=None, help='Path to source file (optional for openclaw)')
@click.option('--glenn', 'format', flag_value='glenn', help='Glenn SQLite format')
@click.option('--claude', 'format', flag_value='claude', help='Claude JSON format')
@click.option('--openclaw', 'format', flag_value='openclaw', help='OpenClaw JSONL sessions')
@click.option('--git', 'format', flag_value='git', help='Git repository history')
@click.option('--limit', '-n', default=None, type=int, help='Limit number of messages to process')
@click.option('--conversation', '-c', default=None, type=int, help='Process specific conversation ID (glenn only)')
@click.option('--user', '-u', default=None, type=str, help='Filter by username (glenn only)')
@click.option('--since', default=None, type=str, help='Import commits since this date (git only, e.g. 2025-01-01)')
@click.option('--parallel', '-p', default=10, type=int, help='Number of parallel workers (default: 10)')
def import_cmd(workspace, db, source, format, limit, conversation, user, since, parallel):
    if not format:
        click.echo('Error: Must specify --glenn, --claude, --openclaw, or --git format')
        return

    if format not in ('openclaw',) and not source:
        if format == 'git':
            click.echo('Error: --source is required for git format (path to git repo)')
        else:
            click.echo('Error: --source is required for glenn and claude formats')
        return

    db_path = get_db_path(workspace, db)
    Path(workspace).mkdir(parents=True, exist_ok=True)
    init_db(str(db_path))

    # Determine format-specific settings
    is_git = format == 'git'
    system_prompt = GIT_OBSERVE_SYSTEM_PROMPT if is_git else None
    user_prompt_prefix = "Extract architectural observations from these git commits:" if is_git else None

    if format == 'glenn':
        messages, info = load_glenn_messages(source, conversation, user, limit)
        if info:
            click.echo(info)
    elif format == 'claude':
        messages, _ = load_claude_messages(source, limit)
    elif format == 'git':
        messages, info = load_git_commits(source, limit=limit, since=since)
        if info:
            click.echo(info)
    else:
        messages, _ = load_openclaw_messages(source, limit)
        file_stats = get_openclaw_file_stats(source)

    source_desc = source or 'default openclaw sessions'
    click.echo(f'Loaded {len(messages)} {"commits" if is_git else "messages"} from {source_desc}')

    if not messages:
        click.echo('No messages to process.')
        return

    by_week = group_messages_by_week(messages)
    weeks = sorted(by_week.keys())
    click.echo(f'Processing {len(weeks)} weeks...')

    total_observations = 0
    session = get_session(str(db_path))

    for week in weeks:
        week_messages = by_week[week]
        click.echo(f'\n{week}: {len(week_messages)} {"commits" if is_git else "messages"}')

        def progress(completed, total, msgs_in_chunk, timestamp, obs_count):
            click.echo(f'  [{completed}/{total}] {obs_count} obs')

        observations = extract_observations(
            week_messages, on_progress=progress, max_workers=parallel,
            system_prompt=system_prompt, user_prompt_prefix=user_prompt_prefix
        )

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

        total_observations += len(observations)
        click.echo(f'  Saved {len(observations)} observations')

    click.echo(f'\nTotal: {total_observations} observations')

    if format == 'openclaw':
        for file_path, (size, mtime) in file_stats.items():
            existing = session.query(ImportedSession).filter_by(file_path=file_path).first()
            if existing:
                existing.last_size = size
                existing.last_mtime = mtime
            else:
                session.add(ImportedSession(file_path=file_path, last_size=size, last_mtime=mtime))
        session.commit()
        click.echo(f'Tracked {len(file_stats)} session files for incremental sync')

    if format == 'git' and messages:
        # Track the last commit hash for incremental sync
        # The last message has the most recent commit info
        last_content = messages[-1]['content']
        last_hash_line = [l for l in last_content.split('\n') if l.startswith('Commit:')]
        if last_hash_line:
            last_hash = last_hash_line[0].split('Commit:')[1].strip()
            # Store using ImportedSession with the repo path as file_path
            existing = session.query(ImportedSession).filter_by(file_path=source).first()
            if existing:
                existing.last_size = 0  # Not meaningful for git, but required
                existing.last_mtime = datetime.now(UTC)
            else:
                session.add(ImportedSession(
                    file_path=source,
                    last_size=0,
                    last_mtime=datetime.now(UTC)
                ))
            session.commit()
            click.echo(f'Tracked last commit: {last_hash}')

    session.close()


@cli.command(help='Sync: import new observations, summarize, embed, synthesize, and write files.')
@click.option('--workspace', '-w', required=True, help='Workspace directory')
@click.option('--db', default='pyramid.db', help='Database filename (default: pyramid.db)')
@click.option('--source', default=None, help='Path to sessions directory for incremental import')
@click.option('--parallel', '-p', default=10, type=int, help='Number of parallel workers (default: 10)')
def sync(workspace, db, source, parallel):
    progress = lambda msg: click.echo(msg)
    do_sync(workspace, db, source, on_progress=progress, max_workers=parallel)


@cli.command('git-sync', help='Sync a git repository: import new commits, summarize, embed, synthesize, write files.')
@click.option('--workspace', '-w', required=True, help='Workspace directory')
@click.option('--db', default='pyramid.db', help='Database filename (default: pyramid.db)')
@click.option('--repo', '-r', required=True, help='Path to git repository')
@click.option('--parallel', '-p', default=10, type=int, help='Number of parallel workers (default: 10)')
def git_sync(workspace, db, repo, parallel):
    from sync import sync as do_sync_full, embed_new_items, write_model_files

    progress = lambda msg: click.echo(msg)
    workspace_path = Path(workspace)
    db_path = workspace_path / db

    workspace_path.mkdir(parents=True, exist_ok=True)
    init_db(str(db_path))

    session = get_session(str(db_path))
    repo_path = str(Path(repo).resolve())

    # Check for last synced commit
    tracking = session.query(ImportedSession).filter_by(file_path=repo_path).first()
    last_hash = None
    if tracking:
        # We need to retrieve the last commit hash — stored in a comment-style field
        # For now, use the git log to determine what's new
        # We'll use the timestamp of the last tracking record
        pass

    messages, new_head, count = load_git_incremental(repo_path, last_hash)

    if not messages:
        click.echo('No new commits to process.')
    else:
        click.echo(f'Found {count} new commits')

        by_week = group_messages_by_week(messages)
        weeks = sorted(by_week.keys())

        total_observations = 0
        for week in weeks:
            week_messages = by_week[week]
            click.echo(f'{week}: {len(week_messages)} commits')

            def obs_progress(completed, total, msgs_in_chunk, timestamp, obs_count):
                click.echo(f'  [{completed}/{total}] {obs_count} obs')

            observations = extract_observations(
                week_messages, on_progress=obs_progress, max_workers=parallel,
                system_prompt=GIT_OBSERVE_SYSTEM_PROMPT,
                user_prompt_prefix="Extract architectural observations from these git commits:"
            )

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
            total_observations += len(observations)

        click.echo(f'Extracted {total_observations} observations')

        # Track the new HEAD
        if tracking:
            tracking.last_mtime = datetime.now(UTC)
        else:
            session.add(ImportedSession(
                file_path=repo_path, last_size=0, last_mtime=datetime.now(UTC)
            ))
        session.commit()

    session.close()

    # Now run the full summarize → embed → synthesize → generate pipeline
    from summarize import run_tier0_summarization, run_higher_tier_summarization, process_all_dirty

    tier0 = run_tier0_summarization(str(db_path), progress, parallel)
    higher = run_higher_tier_summarization(str(db_path), progress, parallel)
    if tier0 or higher:
        click.echo(f'Created {tier0} tier-0 + {higher} higher-tier summaries')

    dirty = process_all_dirty(str(db_path), progress, parallel)
    if dirty:
        click.echo(f'Regenerated {dirty} dirty summaries')

    embedded = embed_new_items(str(db_path), progress, parallel)

    synthesized = synthesize_dirty_models(str(db_path), progress, parallel)

    written = write_model_files(str(db_path), workspace, progress)

    if written:
        click.echo(f'Git sync complete. Updated: {", ".join(written)}')
    else:
        click.echo('Git sync complete.')


@cli.command(help='Semantic search across memory.')
@click.option('--workspace', '-w', required=True, help='Workspace directory')
@click.option('--db', default='pyramid.db', help='Database filename (default: pyramid.db)')
@click.argument('query')
@click.option('--limit', '-n', default=20, help='Number of results to retrieve')
@click.option('--raw', is_flag=True, help='Show raw results without LLM synthesis')
@click.option('--time-weight', '-t', default=0.3, help='Time decay weight (0=pure semantic, 1=heavy recency bias)')
def search(workspace, db, query, limit, raw, time_weight):
    db_path = get_db_path(workspace, db)
    
    if not db_path.exists():
        click.echo(f'Error: No database found at {db_path}')
        click.echo(f'Hint: Run "import" or "observe" first to create the database')
        return
    
    session = get_session(str(db_path))
    conn = sqlite3.connect(str(db_path))
    enable_vec(conn)
    
    results = search_memory(conn, query, limit, time_weight=time_weight)
    
    if not results:
        click.echo('No results. Run "sync" first to create embeddings.')
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


@cli.group(help='Internal commands for debugging and manual control.')
def internal():
    pass


@internal.command('observe', help='Add a single observation manually.')
@click.option('--workspace', '-w', required=True, help='Workspace directory')
@click.option('--db', default='pyramid.db', help='Database filename (default: pyramid.db)')
@click.argument('text')
def observe_cmd(workspace, db, text):
    db_path = get_db_path(workspace, db)
    Path(workspace).mkdir(parents=True, exist_ok=True)
    init_db(str(db_path))
    
    session = get_session(str(db_path))
    obs = Observation(text=text, timestamp=datetime.now(UTC))
    session.add(obs)
    session.commit()
    click.echo(f'Added observation #{obs.id}')
    session.close()


@internal.command('summarize', help='Run summarization to compress observations.')
@click.option('--workspace', '-w', required=True, help='Workspace directory')
@click.option('--db', default='pyramid.db', help='Database filename (default: pyramid.db)')
@click.option('--max-obs', '-n', default=None, type=int, help='Maximum observations to process')
@click.option('--max-tier', '-T', default=None, type=int, help='Maximum tier to build')
@click.option('--parallel', '-p', default=10, type=int, help='Number of parallel workers')
def summarize_cmd(workspace, db, max_obs, max_tier, parallel):
    db_path = get_db_path(workspace, db)
    
    if not db_path.exists():
        click.echo(f'Error: No database found at {db_path}')
        return
    
    progress = lambda msg: click.echo(msg)
    
    click.echo('Running summarization...')
    tier0, higher = run_all_summarization(str(db_path), on_progress=progress, max_workers=parallel, max_tier=max_tier, max_obs=max_obs)
    click.echo(f'Created {tier0} tier 0 + {higher} higher tier summaries')


@internal.command('embed', help='Generate embeddings for semantic search.')
@click.option('--workspace', '-w', required=True, help='Workspace directory')
@click.option('--db', default='pyramid.db', help='Database filename (default: pyramid.db)')
@click.option('--parallel', '-p', default=10, help='Number of parallel workers')
@click.option('--force', is_flag=True, help='Clear existing embeddings and re-embed everything')
def embed_cmd(workspace, db, parallel, force):
    db_path = get_db_path(workspace, db)
    
    if not db_path.exists():
        click.echo(f'Error: No database found at {db_path}')
        return
    
    session = get_session(str(db_path))
    conn = sqlite3.connect(str(db_path))
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


@internal.command('generate', help='Write markdown files from cached synthesis.')
@click.option('--workspace', '-w', required=True, help='Workspace directory')
@click.option('--db', default='pyramid.db', help='Database filename (default: pyramid.db)')
@click.option('--parallel', '-p', default=10, type=int, help='Number of parallel workers')
def generate_cmd(workspace, db, parallel):
    db_path = get_db_path(workspace, db)
    
    if not db_path.exists():
        click.echo(f'Error: No database found at {db_path}')
        return
    
    progress = lambda msg: click.echo(msg)
    regenerated = export_models(workspace, str(db_path), on_progress=progress, max_workers=parallel)
    click.echo(f'Generated: {", ".join(regenerated)}')


@internal.command('synthesize', help='Synthesize dirty models (without writing files).')
@click.option('--workspace', '-w', required=True, help='Workspace directory')
@click.option('--db', default='pyramid.db', help='Database filename (default: pyramid.db)')
@click.option('--parallel', '-p', default=10, type=int, help='Number of parallel workers')
def synthesize_cmd(workspace, db, parallel):
    db_path = get_db_path(workspace, db)
    
    if not db_path.exists():
        click.echo(f'Error: No database found at {db_path}')
        return
    
    progress = lambda msg: click.echo(msg)
    count = synthesize_dirty_models(str(db_path), on_progress=progress, max_workers=parallel)
    click.echo(f'Synthesized {count} models')


if __name__ == '__main__':
    cli()
