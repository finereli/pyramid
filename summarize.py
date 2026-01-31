import json
from datetime import datetime, timedelta, UTC
from sqlalchemy import func
from db import get_session, Observation, Summary, Model, BASE_MODELS
from llm import client, MODEL, MAX_TOKENS, CHARS_PER_TOKEN, estimate_tokens

SPAN = timedelta(days=1)
STEP = 3

SUMMARIZE_SYSTEM_PROMPT = """You are a memory agent creating summaries. Write telegram-style: short, dense sentences capturing the gist.

Mark critical information with IMPORTANT:, CRITICAL:, or ESSENTIAL: prefix.

Example:
"Family: wife Yael 44, kids Tom 11 and Yara 5. IMPORTANT: Currently nomadic since March 2025. Living in Greece temporarily, planning Israel visit."

Be concise. Preserve specific facts: names, dates, numbers, places."""

ASSIGN_MODEL_TOOL = {
    "type": "function", 
    "function": {
        "name": "assign_model",
        "description": "Assign an observation to a mental model",
        "parameters": {
            "type": "object",
            "properties": {
                "observation_id": {"type": "integer"},
                "model_name": {"type": "string", "description": "Model name: self, user, system, or a new topic name"}
            },
            "required": ["observation_id", "model_name"]
        }
    }
}


def get_models_context(session):
    models = session.query(Model).all()
    lines = ["Available models:"]
    for m in models:
        lines.append(f"- {m.name}: {m.description or '(no description)'}")
    return "\n".join(lines)


def assign_models_to_observations(session, observations):
    if not observations:
        return
    
    models_context = get_models_context(session)
    
    obs_text = "\n".join(f"[{obs.id}] {obs.text}" for obs in observations)
    
    prompt = f"""{models_context}

Assign each observation to the most appropriate model. Use 'user' for facts about the primary user, 'self' for agent experiences, 'system' for technical details. Create new models only for major recurring topics.

Observations:
{obs_text}"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You assign observations to mental models. Call assign_model for each observation."},
            {"role": "user", "content": prompt}
        ],
        tools=[ASSIGN_MODEL_TOOL],
        tool_choice="required"
    )
    
    for tool_call in response.choices[0].message.tool_calls or []:
        if tool_call.function.name == "assign_model":
            args = json.loads(tool_call.function.arguments)
            obs_id = args['observation_id']
            model_name = args['model_name'].lower().strip()
            
            model = session.query(Model).filter_by(name=model_name).first()
            if not model:
                model = Model(name=model_name, is_base=False)
                session.add(model)
                session.flush()
            
            obs = session.query(Observation).get(obs_id)
            if obs:
                obs.model_id = model.id
    
    session.commit()


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


def run_tier0_summarization(on_progress=None):
    session = get_session()
    
    unassigned = session.query(Observation).filter(Observation.model_id == None).all()
    if unassigned:
        if on_progress:
            on_progress(f"Assigning {len(unassigned)} observations to models...")
        assign_models_to_observations(session, unassigned)
    
    by_day = get_observations_by_day(session)
    
    created = 0
    for day, observations in sorted(by_day.items()):
        by_model = {}
        for obs in observations:
            model_id = obs.model_id
            if model_id not in by_model:
                by_model[model_id] = []
            by_model[model_id].append(obs)
        
        for model_id, model_obs in by_model.items():
            existing = session.query(Summary).filter(
                Summary.tier == 0,
                Summary.model_id == model_id,
                Summary.start_timestamp == day,
                Summary.end_timestamp == day + SPAN
            ).first()
            
            if existing:
                continue
            
            model = session.query(Model).get(model_id)
            if on_progress:
                on_progress(f"Summarizing {day.date()} [{model.name}] ({len(model_obs)} obs)...")
            
            summary_text = summarize_observations(model_obs)
            
            summary = Summary(
                model_id=model_id,
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
