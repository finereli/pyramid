import json
from datetime import datetime, timedelta, UTC
from sqlalchemy import func
from db import get_session, Observation, Summary, Model, BASE_MODELS
from llm import client, MODEL, MAX_TOKENS, CHARS_PER_TOKEN, estimate_tokens

SPAN = timedelta(days=1)
STEP = 3

SUMMARIZE_SYSTEM_PROMPT = """You are a memory agent creating summaries. Write telegram-style: short, dense sentences.

OUTPUT FORMAT:
- Each line starts with [N] where N is importance (1-10)
- Use IMPORTANT:/CRITICAL:/ESSENTIAL: inline to emphasize key facts within a line
- Example: "[7] User relocated to Austin. CRITICAL: Starting consulting practice May 2025."
- Example: "[5] Family: spouse Sam 38, daughter Mia 7."

Preserve specific facts: names, dates, numbers, places."""

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

Assign each observation to the most appropriate model. Create new models for distinct entities (people, projects, companies) that have multiple observations.

Observations:
{obs_text}"""

    system_prompt = """You assign observations to mental models. Call assign_model for each observation.

Base models (use these when appropriate):
- self: Agent capabilities, preferences, tools used, learnings about being an assistant
- user: Primary user identity, preferences, personal life events, general user info
- system: Technical environment, configurations, software setup, tools

Create new models for distinct entities (specific people, projects, companies, topics) that warrant separate tracking."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        tools=[ASSIGN_MODEL_TOOL],
        tool_choice="required"
    )
    
    for tool_call in response.choices[0].message.tool_calls or []:
        if tool_call.function.name == "assign_model":
            args = json.loads(tool_call.function.arguments)
            obs_id = args['observation_id']
            model_name = args['model_name'].lower().strip().replace(' ', '-')
            
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
            {"role": "user", "content": f"Combine these summaries into one. Keep the [N] importance format:\n\n{combined}"}
        ]
    )
    return response.choices[0].message.content


def summarize_chunk(observations):
    obs_text = "\n".join(f"[{obs.importance}] {obs.text}" for obs in observations)
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Summarize these observations. Each has [importance] prefix (1-10 scale). Preserve importance in output using same [N] format:\n\n{obs_text}"}
        ]
    )
    return response.choices[0].message.content


def summarize_summaries(summaries):
    text = "\n---\n".join(s.text for s in summaries)
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT + "\n\nPreserve [N] importance prefixes and IMPORTANT/CRITICAL/ESSENTIAL markers from source summaries."},
            {"role": "user", "content": f"Combine these summaries into one higher-level summary. Keep the [N] importance format:\n\n{text}"}
        ]
    )
    return response.choices[0].message.content


def run_tier0_summarization(on_progress=None, max_workers=10):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    session = get_session()
    
    unassigned = session.query(Observation).filter(Observation.model_id == None).all()
    if unassigned:
        if on_progress:
            on_progress(f"Assigning {len(unassigned)} observations to models...")
        assign_models_to_observations(session, unassigned)
    
    by_day = get_observations_by_day(session)
    
    tasks = []
    for day, observations in sorted(by_day.items()):
        by_model = {}
        for obs in observations:
            model_id = obs.model_id
            if model_id not in by_model:
                by_model[model_id] = []
            by_model[model_id].append(obs)
        
        for model_id, model_obs in by_model.items():
            if model_id is None:
                continue
            
            existing = session.query(Summary).filter(
                Summary.tier == 0,
                Summary.model_id == model_id,
                Summary.start_timestamp == day,
                Summary.end_timestamp == day + SPAN
            ).first()
            
            if existing:
                continue
            
            model = session.query(Model).get(model_id)
            tasks.append((day, model_id, model.name, model_obs))
    
    if not tasks:
        session.close()
        return 0
    
    if on_progress:
        on_progress(f"Summarizing {len(tasks)} day/model combinations...")
    
    def process_task(task):
        day, model_id, model_name, model_obs = task
        summary_text = summarize_observations(model_obs)
        return (day, model_id, summary_text)
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_task, task): task for task in tasks}
        for future in as_completed(futures):
            results.append(future.result())
    
    for day, model_id, summary_text in results:
        summary = Summary(
            model_id=model_id,
            tier=0,
            text=summary_text,
            start_timestamp=day,
            end_timestamp=day + SPAN
        )
        session.add(summary)
    
    session.commit()
    session.close()
    return len(results)


def run_higher_tier_summarization(on_progress=None, max_workers=10):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    session = get_session()
    models = session.query(Model).all()
    
    total_created = 0
    tier = 0
    
    while True:
        tasks = []
        
        for model in models:
            summaries = session.query(Summary).filter(
                Summary.model_id == model.id,
                Summary.tier == tier
            ).order_by(Summary.start_timestamp).all()
            
            unsummarized = []
            for s in summaries:
                already_summarized = session.query(Summary).filter(
                    Summary.model_id == model.id,
                    Summary.tier == tier + 1,
                    Summary.start_timestamp <= s.start_timestamp,
                    Summary.end_timestamp >= s.end_timestamp
                ).first()
                if not already_summarized:
                    unsummarized.append(s)
            
            if len(unsummarized) < STEP:
                continue
            
            for i in range(0, len(unsummarized) - STEP + 1, STEP):
                chunk = unsummarized[i:i + STEP]
                start_ts = chunk[0].start_timestamp
                end_ts = chunk[-1].end_timestamp
                
                existing = session.query(Summary).filter(
                    Summary.model_id == model.id,
                    Summary.tier == tier + 1,
                    Summary.start_timestamp == start_ts,
                    Summary.end_timestamp == end_ts
                ).first()
                
                if existing:
                    continue
                
                chunk_texts = [s.text for s in chunk]
                tasks.append((model.id, model.name, tier + 1, start_ts, end_ts, chunk_texts))
        
        if not tasks:
            break
        
        if on_progress:
            on_progress(f"T{tier + 1}: {len(tasks)} summaries across models...")
        
        def process_task(task):
            model_id, model_name, new_tier, start_ts, end_ts, chunk_texts = task
            
            class FakeSummary:
                def __init__(self, text):
                    self.text = text
            
            fake_summaries = [FakeSummary(t) for t in chunk_texts]
            summary_text = summarize_summaries(fake_summaries)
            return (model_id, new_tier, start_ts, end_ts, summary_text)
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_task, task): task for task in tasks}
            for future in as_completed(futures):
                results.append(future.result())
        
        for model_id, new_tier, start_ts, end_ts, summary_text in results:
            summary = Summary(
                model_id=model_id,
                tier=new_tier,
                text=summary_text,
                start_timestamp=start_ts,
                end_timestamp=end_ts
            )
            session.add(summary)
            total_created += 1
        
        session.commit()
        tier += 1
    
    session.close()
    return total_created


def run_all_summarization(on_progress=None):
    tier0_count = run_tier0_summarization(on_progress)
    higher_count = run_higher_tier_summarization(on_progress)
    return tier0_count, higher_count
