import os
import struct
import math
from datetime import datetime, UTC
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite_vec
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Configurable embedding backend via environment variables:
#   PYRAMID_EMBEDDING_BASE_URL  — OpenAI-compatible API base (default: OpenAI)
#   PYRAMID_EMBEDDING_API_KEY   — API key (falls back to PYRAMID_LLM_API_KEY, then OPENAI_API_KEY)
#   PYRAMID_EMBEDDING_MODEL     — Model name (default: text-embedding-3-small)
#   PYRAMID_EMBEDDING_DIM       — Embedding dimensions (default: 1536)
#
# Examples:
#   OpenAI (default):  OPENAI_API_KEY=sk-...
#   Ollama:            PYRAMID_EMBEDDING_BASE_URL=http://localhost:11434/v1 PYRAMID_EMBEDDING_MODEL=nomic-embed-text
#   Voyage AI:         PYRAMID_EMBEDDING_BASE_URL=https://api.voyageai.com/v1 PYRAMID_EMBEDDING_API_KEY=pa-...

_embed_kwargs = {}
_base_url = os.environ.get('PYRAMID_EMBEDDING_BASE_URL')
_api_key = os.environ.get('PYRAMID_EMBEDDING_API_KEY') or os.environ.get('PYRAMID_LLM_API_KEY')

if _base_url:
    _embed_kwargs['base_url'] = _base_url
if _api_key:
    _embed_kwargs['api_key'] = _api_key

client = OpenAI(**_embed_kwargs)
EMBEDDING_MODEL = os.environ.get('PYRAMID_EMBEDDING_MODEL', 'text-embedding-3-small')
EMBEDDING_DIM = int(os.environ.get('PYRAMID_EMBEDDING_DIM', '1536'))

TIME_DECAY_HALF_LIFE_DAYS = 30


def format_temporal_prefix(timestamp, end_timestamp=None):
    if timestamp is None:
        return ""
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    if end_timestamp:
        if isinstance(end_timestamp, str):
            end_timestamp = datetime.fromisoformat(end_timestamp.replace('Z', '+00:00'))
        if timestamp.year == end_timestamp.year and timestamp.month == end_timestamp.month:
            return f"In {timestamp.strftime('%B %Y')}: "
        elif timestamp.year == end_timestamp.year:
            return f"From {timestamp.strftime('%B')} to {end_timestamp.strftime('%B %Y')}: "
        else:
            return f"From {timestamp.strftime('%B %Y')} to {end_timestamp.strftime('%B %Y')}: "
    return f"In {timestamp.strftime('%B %Y')}: "


def enrich_for_embedding(text, timestamp, end_timestamp=None):
    prefix = format_temporal_prefix(timestamp, end_timestamp)
    return f"{prefix}{text}"


MAX_TOKENS_PER_REQUEST = 250000
MAX_ITEMS_PER_REQUEST = 2048
CHARS_PER_TOKEN = 4


def estimate_tokens(text):
    return len(text) // CHARS_PER_TOKEN


def get_embedding(text):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def get_embeddings_batch(texts):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]


def batch_by_tokens(texts, max_tokens=MAX_TOKENS_PER_REQUEST, max_items=MAX_ITEMS_PER_REQUEST):
    batches = []
    current_batch = []
    current_tokens = 0
    
    for text in texts:
        tokens = estimate_tokens(text)
        if (current_tokens + tokens > max_tokens or len(current_batch) >= max_items) and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(text)
        current_tokens += tokens
    
    if current_batch:
        batches.append(current_batch)
    
    return batches


def embed_many(texts, max_workers=10, on_progress=None):
    batches = batch_by_tokens(texts)
    results = [None] * len(batches)
    items_done = 0
    batches_done = 0
    
    def process_batch(batch_idx):
        return batch_idx, get_embeddings_batch(batches[batch_idx])
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_batch, i) for i in range(len(batches))]
        for future in as_completed(futures):
            batch_idx, embeddings = future.result()
            results[batch_idx] = embeddings
            items_done += len(embeddings)
            batches_done += 1
            if on_progress:
                on_progress(items_done, len(texts), batches_done, len(batches))
    
    return [emb for batch in results for emb in batch]


def serialize_embedding(embedding):
    return struct.pack(f'{len(embedding)}f', *embedding)


def deserialize_embedding(blob):
    count = len(blob) // 4
    return list(struct.unpack(f'{count}f', blob))


def enable_vec(conn):
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def create_vec_table(conn, table_name, dim=EMBEDDING_DIM):
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {table_name}_vec USING vec0(
            embedding float[{dim}]
        )
    """)


def search_similar(conn, table_name, query_embedding, limit=10):
    query_blob = serialize_embedding(query_embedding)
    
    results = conn.execute(f"""
        SELECT rowid, distance
        FROM {table_name}_vec
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
    """, [query_blob, limit]).fetchall()
    
    return results


def init_memory_vec(conn):
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0(
            id INTEGER PRIMARY KEY,
            source_type TEXT,
            source_id INTEGER,
            embedding float[{EMBEDDING_DIM}]
        )
    """)


def get_existing_embeddings(conn):
    return set(
        (row[0], row[1]) 
        for row in conn.execute("SELECT source_type, source_id FROM memory_vec").fetchall()
    )


def store_embeddings(conn, items, embeddings):
    for i, (source_type, source_id, _) in enumerate(items):
        conn.execute(
            "INSERT INTO memory_vec (source_type, source_id, embedding) VALUES (?, ?, ?)",
            [source_type, source_id, serialize_embedding(embeddings[i])]
        )
    conn.commit()


def compute_time_penalty(timestamp, half_life_days=TIME_DECAY_HALF_LIFE_DAYS):
    if timestamp is None:
        return 0.5
    now = datetime.now(UTC)
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    age_days = (now - timestamp).total_seconds() / 86400
    decay_rate = math.log(2) / half_life_days
    return 1 - math.exp(-age_days * decay_rate)


def search_memory(conn, query_text, limit=20, time_weight=0.3):
    query_embedding = get_embedding(query_text)
    fetch_limit = limit * 3 if time_weight > 0 else limit
    
    candidates = conn.execute("""
        SELECT source_type, source_id, distance
        FROM memory_vec
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
    """, [serialize_embedding(query_embedding), fetch_limit]).fetchall()
    
    if time_weight == 0:
        return candidates
    
    obs_ids = [r[1] for r in candidates if r[0] == 'observation']
    sum_ids = [r[1] for r in candidates if r[0] == 'summary']
    
    timestamps = {}
    if obs_ids:
        placeholders = ','.join('?' * len(obs_ids))
        rows = conn.execute(
            f"SELECT id, timestamp FROM observations WHERE id IN ({placeholders})",
            obs_ids
        ).fetchall()
        for id, ts in rows:
            timestamps[('observation', id)] = ts
    
    if sum_ids:
        placeholders = ','.join('?' * len(sum_ids))
        rows = conn.execute(
            f"SELECT id, end_timestamp FROM summaries WHERE id IN ({placeholders})",
            sum_ids
        ).fetchall()
        for id, ts in rows:
            timestamps[('summary', id)] = ts
    
    scored = []
    for source_type, source_id, distance in candidates:
        ts = timestamps.get((source_type, source_id))
        time_penalty = compute_time_penalty(ts)
        final_score = distance * (1 - time_weight) + time_penalty * time_weight
        scored.append((source_type, source_id, distance, final_score))
    
    scored.sort(key=lambda x: x[3])
    return [(r[0], r[1], r[2]) for r in scored[:limit]]
