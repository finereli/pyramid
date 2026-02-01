from db import Summary, Observation
from llm import client, MODEL


def get_pyramid(session, model_id):
    summaries = session.query(Summary).filter_by(model_id=model_id)\
        .order_by(Summary.tier.desc(), Summary.start_timestamp.desc()).all()
    
    by_tier = {}
    for s in summaries:
        by_tier.setdefault(s.tier, []).append(s)
    return by_tier


def get_unsummarized_observations(session, model_id, by_tier):
    tier0_summaries = by_tier.get(0, [])
    if not tier0_summaries:
        return session.query(Observation).filter(
            Observation.model_id == model_id
        ).order_by(Observation.timestamp).all()
    
    last_summarized_ts = max(s.end_timestamp for s in tier0_summaries)
    return session.query(Observation).filter(
        Observation.model_id == model_id,
        Observation.timestamp > last_summarized_ts
    ).order_by(Observation.timestamp).all()


def synthesize_model(name, description, by_tier, unsummarized_obs=None):
    parts = []
    
    for tier in sorted(by_tier.keys(), reverse=True):
        for s in sorted(by_tier[tier], key=lambda x: x.end_timestamp, reverse=True):
            parts.append(f"[{s.end_timestamp:%Y-%m-%d}] {s.text}")
    
    if unsummarized_obs:
        obs_section = "\n".join(f"- {o.text}" for o in unsummarized_obs)
        parts.append(f"\n[Recent unsummarized observations]\n{obs_section}")
    
    if not parts:
        return None
    
    prompt = f"""Synthesize this information about '{name}' into a coherent mental model.

Model purpose: {description or 'Not specified'}

Content is listed newest first. Rules:
- Newer details override older ones (e.g., if age changes, use the most recent)
- Deduplicate repeated facts - mention each fact once
- Preserve important historical context and evolution
- Write in third person narrative prose
- Be comprehensive but not repetitive

Content:
{chr(10).join(parts)}

Write a synthesized model of '{name}':"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
