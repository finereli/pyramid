import sqlite3
from pathlib import Path
from datetime import datetime, UTC

from db import init_db, get_session, Observation, Model, Summary, ImportedSession
from llm import extract_observations
from loaders import load_openclaw_incremental, get_openclaw_file_stats, group_messages_by_week
from summarize import (
    run_tier0_summarization,
    run_higher_tier_summarization,
    process_all_dirty,
    mark_model_dirty,
    mark_overlapping_summaries_dirty
)
from pyramid import synthesize_dirty_models
from embeddings import (
    embed_many,
    enable_vec,
    init_memory_vec,
    get_existing_embeddings,
    store_embeddings,
    enrich_for_embedding
)


def embed_new_items(db_path, on_progress=None, max_workers=10):
    session = get_session(db_path)
    conn = sqlite3.connect(db_path)
    enable_vec(conn)
    init_memory_vec(conn)
    
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
        conn.close()
        session.close()
        return 0
    
    if on_progress:
        on_progress(f"Embedding {len(to_embed)} items...")
    
    texts = [item[2] for item in to_embed]
    embeddings = embed_many(texts, max_workers=max_workers)
    
    store_embeddings(conn, to_embed, embeddings)
    conn.close()
    session.close()
    
    if on_progress:
        on_progress(f"Embedded {len(to_embed)} items")
    
    return len(to_embed)


def write_model_files(db_path, workspace, on_progress=None):
    from generate import export_models
    return export_models(workspace, db_path, on_progress=on_progress)


def sync(workspace, db='pyramid.db', source=None, on_progress=None, max_workers=10):
    workspace = Path(workspace)
    db_path = workspace / db
    
    workspace.mkdir(parents=True, exist_ok=True)
    init_db(str(db_path))
    
    session = get_session(str(db_path))
    
    if source:
        tracking = {}
        for rec in session.query(ImportedSession).all():
            tracking[rec.file_path] = (rec.last_size, rec.last_mtime)
        
        messages, updated_tracking, changed_files = load_openclaw_incremental(source, tracking)
        
        if changed_files:
            if on_progress:
                on_progress(f"Found {len(changed_files)} changed files, {len(messages)} new messages")
            
            if messages:
                by_week = group_messages_by_week(messages)
                weeks = sorted(by_week.keys())
                
                total_observations = 0
                for week in weeks:
                    week_messages = by_week[week]
                    if on_progress:
                        on_progress(f"Processing {week}: {len(week_messages)} messages")
                    
                    def progress(completed, total, msgs_in_chunk, timestamp, obs_count):
                        if on_progress:
                            on_progress(f"  [{completed}/{total}] {obs_count} obs")
                    
                    observations = extract_observations(week_messages, on_progress=progress, max_workers=max_workers)
                    
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
                
                if on_progress:
                    on_progress(f"Extracted {total_observations} observations")
            
            for file_path, (size, mtime) in updated_tracking.items():
                existing = session.query(ImportedSession).filter_by(file_path=file_path).first()
                if existing:
                    existing.last_size = size
                    existing.last_mtime = mtime
                else:
                    session.add(ImportedSession(file_path=file_path, last_size=size, last_mtime=mtime))
            session.commit()
        else:
            if on_progress:
                on_progress("No changes detected in source files")
    
    session.close()
    
    tier0 = run_tier0_summarization(str(db_path), on_progress, max_workers)
    higher = run_higher_tier_summarization(str(db_path), on_progress, max_workers)
    if (tier0 or higher) and on_progress:
        on_progress(f"Created {tier0} tier-0 + {higher} higher-tier summaries")
    
    dirty_processed = process_all_dirty(str(db_path), on_progress, max_workers)
    if dirty_processed and on_progress:
        on_progress(f"Regenerated {dirty_processed} dirty summaries")
    
    try:
        embedded = embed_new_items(str(db_path), on_progress, max_workers)
    except Exception as e:
        if on_progress:
            on_progress(f"Skipping embeddings ({e.__class__.__name__}: {e})")
        embedded = 0
    
    synthesized = synthesize_dirty_models(str(db_path), on_progress, max_workers)
    
    written = write_model_files(str(db_path), workspace, on_progress)
    
    if on_progress:
        if written:
            on_progress(f"Sync complete. Updated: {', '.join(written)}")
        elif tier0 or higher or dirty_processed or synthesized:
            on_progress("Sync complete.")
        else:
            on_progress("Nothing to update.")
    
    return written
