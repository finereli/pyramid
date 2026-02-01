import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MODEL = 'gpt-4.1-mini'
MAX_TOKENS = 10000
CHARS_PER_TOKEN = 4

client = OpenAI()

OBSERVE_TOOL = {
    "type": "function",
    "function": {
        "name": "add_observation",
        "description": "Record an observation about the conversation",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The observation text, a single factual sentence"
                }
            },
            "required": ["text"]
        }
    }
}

OBSERVE_SYSTEM_PROMPT = """You are a memory agent extracting facts from conversations.

Capture SPECIFIC DETAILS, not meta-observations.

BAD: "The user shared family information"
GOOD: "User's son Tom is 8 years old"

BAD: "We discussed their preferences"  
GOOD: "User prefers dark mode in all apps"

Include names, places, dates, numbers, preferences. Each observation should be a single factual sentence."""


def estimate_tokens(text):
    return len(text) // CHARS_PER_TOKEN


def chunk_messages(messages, max_tokens=MAX_TOKENS):
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for msg in messages:
        msg_text = f"{msg['role']}: {msg['content']}"
        msg_tokens = estimate_tokens(msg_text)
        
        if current_tokens + msg_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0
        
        current_chunk.append(msg)
        current_tokens += msg_tokens
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def process_chunk(chunk):
    chunk_timestamp = chunk[-1].get('timestamp') if chunk else None
    conversation_text = "\n".join(f"{m['role']}: {m['content']}" for m in chunk if m['content'])
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": OBSERVE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract observations from this conversation:\n\n{conversation_text}"}
        ],
        tools=[OBSERVE_TOOL],
        tool_choice="auto"
    )
    
    observations = []
    for tool_call in response.choices[0].message.tool_calls or []:
        if tool_call.function.name == "add_observation":
            args = json.loads(tool_call.function.arguments)
            args['timestamp'] = chunk_timestamp
            observations.append(args)
    
    return observations, chunk_timestamp


def extract_observations(messages, on_progress=None, max_workers=10):
    chunks = chunk_messages(messages)
    total_chunks = len(chunks)
    
    if total_chunks == 0:
        return []
    
    all_observations = [None] * total_chunks
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_chunk, chunk): i for i, chunk in enumerate(chunks)}
        completed = 0
        
        for future in as_completed(futures):
            i = futures[future]
            observations, chunk_timestamp = future.result()
            all_observations[i] = observations
            completed += 1
            
            if on_progress:
                on_progress(completed, total_chunks, len(chunks[i]), chunk_timestamp, len(observations))
    
    return [obs for chunk_obs in all_observations for obs in chunk_obs]
