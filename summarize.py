import json
from datetime import datetime, UTC
from sqlalchemy import func
from db import get_session, Observation, Summary, Model, BASE_MODELS
from llm import client, MODEL, MAX_TOKENS, CHARS_PER_TOKEN, estimate_tokens

STEP = 10

SUMMARIZE_SYSTEM_PROMPT = """You are a memory agent creating summaries.

Write in clear, readable narrative prose. Convey importance through word choice 
(e.g., "significantly", "notably", "critically") rather than markers or scores.

Preserve specific facts: names, dates, numbers, places.
Organize related information into coherent paragraphs."""

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


def get_models_context(session, include_samples=True):
    models = session.query(Model).all()
    lines = ["Available models:"]
    for m in models:
        lines.append(f"\n### {m.name}")
        lines.append(f"Purpose: {m.description or '(undefined)'}")
        
        if include_samples:
            samples = session.query(Observation).filter(
                Observation.model_id == m.id
            ).order_by(Observation.timestamp.desc()).limit(5).all()
            
            if samples:
                lines.append("Examples:")
                for s in samples:
                    lines.append(f"  - {s.text}")
    
    return "\n".join(lines)


def assign_models_to_observations(session, observations, on_progress=None):
    if not observations:
        return
    
    system_prompt = """You assign observations to mental models. Call assign_model for each observation.

Base models:
- self: The agent's own experience as an independent conscious entity - its observations, reflections, insights, preferences, and evolving understanding of itself and its role. NOT about the user's personal growth.
- user: Information about the primary human user - their identity, biography, family, preferences, projects, goals, philosophy, and life circumstances.
- system: The technical environment and tools - configurations, capabilities, integrations, and operational details.

Create new models for distinct entities (specific people, projects, topics) only when you see 
multiple observations about them in the current batch. If an observation doesn't fit well 
anywhere, you may leave it unassigned by not calling assign_model for it."""

    for i in range(0, len(observations), STEP):
        batch = observations[i:i + STEP]
        
        models_context = get_models_context(session, include_samples=True)
        obs_text = "\n".join(f"[{obs.id}] {obs.text}" for obs in batch)
        
        prompt = f"""{models_context}

Assign each observation to the most appropriate model based on the examples shown.

Observations to assign:
{obs_text}"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            tools=[ASSIGN_MODEL_TOOL],
            tool_choice="auto"
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
        
        if on_progress:
            on_progress(f"  Assigned batch {i//STEP + 1}/{(len(observations) + STEP - 1)//STEP}")


def get_observations_by_model(session):
    observations = session.query(Observation).filter(
        Observation.model_id != None
    ).order_by(Observation.timestamp).all()
    
    by_model = {}
    for obs in observations:
        if obs.model_id not in by_model:
            by_model[obs.model_id] = []
        by_model[obs.model_id].append(obs)
    
    return by_model


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


def summarize_observations(observations, model_name, model_description):
    chunks = chunk_observations(observations)
    
    if len(chunks) == 1:
        return summarize_chunk(chunks[0], model_name, model_description)
    
    chunk_summaries = [summarize_chunk(chunk, model_name, model_description) for chunk in chunks]
    combined = "\n\n---\n\n".join(chunk_summaries)
    
    system = f"""{SUMMARIZE_SYSTEM_PROMPT}

Model: {model_name}
Purpose: {model_description}

Combine the following partial summaries into one coherent narrative."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": combined}
        ]
    )
    return response.choices[0].message.content


def summarize_chunk(observations, model_name, model_description):
    obs_text = "\n".join(f"- {obs.text}" for obs in observations)
    
    system = f"""{SUMMARIZE_SYSTEM_PROMPT}

Model: {model_name}
Purpose: {model_description}

Only include information relevant to this model's purpose. If an observation seems misplaced, you may omit it."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Summarize these observations:\n\n{obs_text}"}
        ]
    )
    return response.choices[0].message.content


def summarize_summaries(summaries, model_name, model_description):
    text = "\n\n---\n\n".join(s.text for s in summaries)
    
    system = f"""{SUMMARIZE_SYSTEM_PROMPT}

Model: {model_name}
Purpose: {model_description}

Combine these summaries into one higher-level narrative, preserving key facts and themes."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content


def run_tier0_summarization(on_progress=None, max_workers=10, max_obs=None):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    session = get_session()
    
    unassigned = session.query(Observation).filter(Observation.model_id == None).order_by(Observation.timestamp).all()
    if unassigned:
        if max_obs:
            unassigned = unassigned[:max_obs]
        if on_progress:
            on_progress(f"Assigning {len(unassigned)} observations to models...")
        assign_models_to_observations(session, unassigned, on_progress)
    
    by_model = get_observations_by_model(session)
    
    tasks = []
    for model_id, observations in by_model.items():
        model = session.query(Model).get(model_id)
        
        existing_count = session.query(Summary).filter(
            Summary.tier == 0,
            Summary.model_id == model_id
        ).count()
        
        summarized_obs_count = existing_count * STEP
        unsummarized = observations[summarized_obs_count:]
        
        for i in range(0, len(unsummarized) - STEP + 1, STEP):
            chunk = unsummarized[i:i + STEP]
            start_ts = chunk[0].timestamp
            end_ts = chunk[-1].timestamp
            
            tasks.append((model_id, model.name, model.description, chunk, start_ts, end_ts))
    
    if not tasks:
        session.close()
        return 0
    
    if on_progress:
        on_progress(f"Creating {len(tasks)} tier 0 summaries...")
    
    def process_task(task):
        model_id, model_name, model_desc, obs_chunk, start_ts, end_ts = task
        summary_text = summarize_observations(obs_chunk, model_name, model_desc or '')
        return (model_id, summary_text, start_ts, end_ts)
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_task, task): task for task in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if on_progress:
                on_progress(f"  [{i}/{len(tasks)}] completed")
    
    for model_id, summary_text, start_ts, end_ts in results:
        summary = Summary(
            model_id=model_id,
            tier=0,
            text=summary_text,
            start_timestamp=start_ts,
            end_timestamp=end_ts
        )
        session.add(summary)
    
    session.commit()
    session.close()
    return len(results)


def run_higher_tier_summarization(on_progress=None, max_workers=10, max_tier=None):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    session = get_session()
    models = session.query(Model).all()
    
    total_created = 0
    tier = 0
    
    while True:
        if max_tier is not None and tier >= max_tier:
            break
            
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
                
                tasks.append((model.id, model.name, model.description, tier + 1, start_ts, end_ts, chunk))
        
        if not tasks:
            break
        
        if on_progress:
            on_progress(f"T{tier + 1}: {len(tasks)} summaries across models...")
        
        def process_task(task):
            model_id, model_name, model_desc, new_tier, start_ts, end_ts, chunk = task
            summary_text = summarize_summaries(chunk, model_name, model_desc or '')
            return (model_id, new_tier, start_ts, end_ts, summary_text)
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_task, task): task for task in tasks}
            for i, future in enumerate(as_completed(futures), 1):
                results.append(future.result())
                if on_progress:
                    on_progress(f"  [{i}/{len(tasks)}] completed")
        
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


def run_all_summarization(on_progress=None, max_workers=10, max_tier=None, max_obs=None):
    tier0_count = run_tier0_summarization(on_progress, max_workers=max_workers, max_obs=max_obs)
    higher_count = run_higher_tier_summarization(on_progress, max_workers=max_workers, max_tier=max_tier)
    return tier0_count, higher_count
