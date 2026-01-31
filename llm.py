import os
import json
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
        "description": "Record an observation about the conversation. Write in first person from the agent's perspective.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The observation text, a single sentence in first person"
                },
                "importance": {
                    "type": "integer",
                    "description": "Importance 1-10. 1=trivial, 5=normal, 7=notable, 10=critical"
                }
            },
            "required": ["text", "importance"]
        }
    }
}

OBSERVE_SYSTEM_PROMPT = """You are a memory agent reflecting on conversations. Your job is to extract observations from the conversation provided.

Observations are:
- First person ("I learned...", "Eli mentioned...", "We discussed...")
- Single sentences capturing one idea
- About the agent (self), the user, or topics discussed

Rate importance:
- 1-3: Trivial, routine
- 4-6: Normal, useful context  
- 7-8: Notable, worth remembering
- 9-10: Critical, essential to remember

Call add_observation for each distinct observation. Extract multiple observations from the conversation."""


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


def extract_observations(messages):
    chunks = chunk_messages(messages)
    all_observations = []
    
    for chunk in chunks:
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
        
        for tool_call in response.choices[0].message.tool_calls or []:
            if tool_call.function.name == "add_observation":
                args = json.loads(tool_call.function.arguments)
                all_observations.append(args)
    
    return all_observations
