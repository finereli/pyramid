import json
from datetime import datetime, timedelta, UTC
from sqlalchemy import func
from db import get_session, Observation, Summary, Model
from llm import client, MODEL, MAX_TOKENS, CHARS_PER_TOKEN, estimate_tokens

SPAN = timedelta(days=1)
STEP = 3

SUMMARIZE_SYSTEM_PROMPT = """You are a memory agent creating summaries. Write telegram-style: short, dense sentences capturing the gist.

Mark critical information with IMPORTANT:, CRITICAL:, or ESSENTIAL: prefix.

Example:
"Family: wife Yael 44, kids Tom 11 and Yara 5. IMPORTANT: Currently nomadic since March 2025. Living in Greece temporarily, planning Israel visit."

Be concise. Preserve specific facts: names, dates, numbers, places."""


def get_day_start(dt):
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def get_observations_by_day(session):
    observations = session.query(Observation).order_by(Observation.timestamp).all()
    
    by_day = {}
    for obs in observations:
        day = get_day_start(obs.timestamp)
        if day not in by_day:
            by_day[day] = []
        by_day[day].append(obs)
    
    return by_day


def chunk_observations(observations, max_tokens=MAX_TOKENS):
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for obs in observations:
        obs_tokens = estimate_tokens(obs.text)
        if current_tokens + obs_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0
        current_chunk.append(obs)
        current_tokens += obs_tokens
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def summarize_observations(observations):
    chunks = chunk_observations(observations)
    
    if len(chunks) == 1:
        return summarize_chunk(chunks[0])
    
    chunk_summaries = [summarize_chunk(chunk) for chunk in chunks]
    combined = "\n".join(chunk_summaries)
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Combine these summaries into one:\n\n{combined}"}
        ]
    )
    return response.choices[0].message.content


def summarize_chunk(observations):
    obs_text = "\n".join(f"[{obs.importance}] {obs.text}" for obs in observations)
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Summarize these observations (number is importance 1-10):\n\n{obs_text}"}
        ]
    )
    return response.choices[0].message.content


def run_tier0_summarization():
    session = get_session()
    by_day = get_observations_by_day(session)
    
    created = 0
    for day, observations in sorted(by_day.items()):
        existing = session.query(Summary).filter(
            Summary.tier == 0,
            Summary.start_timestamp == day,
            Summary.end_timestamp == day + SPAN
        ).first()
        
        if existing:
            continue
        
        summary_text = summarize_observations(observations)
        
        default_model = session.query(Model).filter_by(name='self').first()
        summary = Summary(
            model_id=default_model.id,
            tier=0,
            text=summary_text,
            start_timestamp=day,
            end_timestamp=day + SPAN
        )
        session.add(summary)
        created += 1
    
    session.commit()
    session.close()
    return created
