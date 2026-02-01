from datetime import timedelta
from db import Summary, Observation
from llm import client, MODEL


TIME_BUCKETS = [
    ('Last 3 Days', timedelta(days=3)),
    ('This Week', timedelta(days=7)),
    ('This Month', timedelta(days=30)),
    ('This Quarter', timedelta(days=90)),
    ('This Year', timedelta(days=365)),
    ('Earlier', None),
]


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


def bucket_by_time(items, ref_date):
    buckets = {label: [] for label, _ in TIME_BUCKETS}
    prev_cutoff = ref_date
    
    for label, delta in TIME_BUCKETS:
        if delta is None:
            cutoff = None
        else:
            cutoff = ref_date - delta
        
        for item in items:
            ts = item['end_timestamp'] if 'end_timestamp' in item else item.get('timestamp')
            if ts is None:
                continue
            if cutoff is None:
                if ts < prev_cutoff:
                    buckets[label].append(item)
            elif cutoff < ts <= prev_cutoff:
                buckets[label].append(item)
        
        prev_cutoff = cutoff
    
    return buckets


def synthesize_model(name, description, by_tier, unsummarized_obs=None, ref_date=None):
    all_items = []
    
    for tier in sorted(by_tier.keys(), reverse=True):
        for s in by_tier[tier]:
            all_items.append({
                'end_timestamp': s['end_timestamp'],
                'text': s['text'],
                'tier': tier
            })
    
    if unsummarized_obs:
        for text, ts in unsummarized_obs:
            all_items.append({
                'end_timestamp': ts or ref_date,
                'text': text,
                'tier': -1
            })
    
    if not all_items:
        return None
    
    if ref_date is None:
        ref_date = max(item['end_timestamp'] for item in all_items)
    
    buckets = bucket_by_time(all_items, ref_date)
    
    sections = []
    for label, _ in TIME_BUCKETS:
        items = buckets[label]
        if not items:
            continue
        items_sorted = sorted(items, key=lambda x: x['end_timestamp'], reverse=True)
        content = "\n".join(f"[{i['end_timestamp']:%Y-%m-%d}] {i['text']}" for i in items_sorted)
        sections.append(f"### {label}\n{content}")
    
    if not sections:
        return None
    
    prompt = f"""Synthesize this information about '{name}' into a coherent mental model.

Model purpose: {description or 'Not specified'}
Reference date: {ref_date:%Y-%m-%d}

Content is organized by recency. Rules:
- Output MUST have these sections: Last 3 Days, This Week, This Month, This Quarter, This Year, Earlier
- Only include sections that have content
- Within each section, synthesize and deduplicate the information
- Newer details override older ones (e.g., if age changes, use the most recent)
- Write in third person narrative prose
- Each section should be self-contained but avoid repetition across sections

Content:
{chr(10).join(sections)}

Write a synthesized model of '{name}' with temporal sections:"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
