from datetime import timedelta, datetime, UTC
from concurrent.futures import ThreadPoolExecutor, as_completed
from db import Summary, Observation, Model, get_session
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
    if ref_date.tzinfo is not None:
        ref_date = ref_date.replace(tzinfo=None)
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
            if ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)
            if cutoff is None:
                if ts < prev_cutoff:
                    buckets[label].append(item)
            elif cutoff < ts <= prev_cutoff:
                buckets[label].append(item)
        
        prev_cutoff = cutoff
    
    return buckets


def get_non_overlapping_summaries(by_tier):
    result = []
    higher_tier_max_end = None
    
    for tier in sorted(by_tier.keys(), reverse=True):
        current_tier_max_end = None
        
        for s in sorted(by_tier[tier], key=lambda x: x['end_timestamp'], reverse=True):
            end = s['end_timestamp']
            
            if higher_tier_max_end is None or end > higher_tier_max_end:
                result.append({
                    'end_timestamp': end,
                    'start_timestamp': s.get('start_timestamp') or end,
                    'text': s['text'],
                    'tier': tier
                })
                current_tier_max_end = max(current_tier_max_end or end, end)
        
        if current_tier_max_end:
            higher_tier_max_end = max(higher_tier_max_end or current_tier_max_end, current_tier_max_end)
    
    return result


def synthesize_model(name, description, by_tier, unsummarized_obs=None, ref_date=None):
    all_items = get_non_overlapping_summaries(by_tier)
    
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
    
    if name == 'assistant':
        voice = "first person (I, me, my) as the AI assistant reflecting on my own experience"
    else:
        voice = "dry, factual technical prose — like a senior developer's personal notes, not documentation or marketing. No superlatives, no 'robust', no 'sophisticated', no 'seamlessly'. Just say what it does and how"

    prompt = f"""Synthesize this information about '{name}' into a coherent mental model.

Model purpose: {description or 'Not specified'}
Reference date: {ref_date:%Y-%m-%d}

Content is organized by recency. Rules:
- Output MUST have these sections: Last 3 Days, This Week, This Month, This Quarter, This Year, Earlier
- Only include sections that have content
- Within each section, synthesize and deduplicate the information
- Newer details override older ones (e.g., if a pattern was replaced, describe the current state)
- Write in {voice}
- Each section should be self-contained but avoid repetition across sections
- Focus on HOW things work and WHY they were built that way, not just WHAT exists
- Include specific file names, function names, and endpoints when known
- No summary paragraphs at the end — the content speaks for itself

Content:
{chr(10).join(sections)}

Write a synthesized model of '{name}' with temporal sections:"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def prepare_model_data(session, model, ref_date=None):
    by_tier_raw = get_pyramid(session, model.id)
    unsummarized = get_unsummarized_observations(session, model.id, by_tier_raw)
    
    by_tier = {
        tier: [{'text': s.text, 'start_timestamp': s.start_timestamp, 'end_timestamp': s.end_timestamp} for s in sums]
        for tier, sums in by_tier_raw.items()
    }
    unsummarized_with_ts = [(o.text, o.timestamp) for o in unsummarized]
    
    return {
        'name': model.name,
        'description': model.description,
        'by_tier': by_tier,
        'unsummarized': unsummarized_with_ts,
        'ref_date': ref_date or datetime.now(UTC),
    }


def synthesize_one_model(data):
    if not data['by_tier'] and not data['unsummarized']:
        return None
    return synthesize_model(
        data['name'],
        data['description'],
        data['by_tier'],
        data['unsummarized'],
        data['ref_date']
    )


def synthesize_dirty_models(db_path, on_progress=None, max_workers=10, ref_date=None):
    session = get_session(db_path)
    
    dirty_models = session.query(Model).filter(Model.content_dirty == True).all()
    
    if not dirty_models:
        session.close()
        return 0
    
    if on_progress:
        on_progress(f"Synthesizing {len(dirty_models)} dirty models...")
    
    global_ref_date = ref_date or datetime.now(UTC)
    
    model_data_list = []
    for model in dirty_models:
        data = prepare_model_data(session, model, global_ref_date)
        data['model_id'] = model.id
        model_data_list.append(data)
    
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(synthesize_one_model, data): data for data in model_data_list}
        for i, future in enumerate(as_completed(futures), 1):
            data = futures[future]
            content = future.result()
            results[data['model_id']] = content
            if on_progress:
                on_progress(f"  [{i}/{len(model_data_list)}] {data['name']}")
    
    for model in dirty_models:
        model.synthesized_content = results.get(model.id)
        model.content_dirty = False
    
    session.commit()
    session.close()
    return len(dirty_models)
