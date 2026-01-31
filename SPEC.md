# Pyramid Memory System Specification

## Overview

A pyramidal memory system for AI agents. The agent reflects on information and generates observations, which are compressed into tiered summaries organized around mental models.

## Core Concepts

### Observations

- Short text strings (usually single sentences) from the agent's first-person perspective
- Stored with timestamp and importance (1-10+ scale)
- Initially unassigned to any model (model_id=NULL)
- Created via LLM tool calls: `add_observation(text, importance)`

### Models

Mental models about concepts - self, users, people, places, topics.

**Base models (always present):**
- `self` - the agent's experiences, capabilities, preferences
- `user` - the primary user (Eli)
- `system` - the technical environment

**Discovered models:** Created during summarization when patterns emerge.

Each model has:
- `name` - unique identifier
- `description` - derived from top-level summary
- `is_base` - true for self/user/system

### Summaries

Telegram-style compressed observations organized by tier.

**Constants:**
- `SPAN = 1 day` - base time unit
- `STEP = 3` - items per summary

**Tier structure:**
- Tier 0: Summarizes 1 day of observations
- Tier 1: Summarizes 3 tier-0 summaries (3 days)
- Tier 2: Summarizes 3 tier-1 summaries (9 days)
- Tier N: Summarizes 3 tier-(N-1) summaries (3^N days)

**Importance markers in summaries:** IMPORTANT, CRITICAL, ESSENTIAL

### Pyramid Retrieval

For any model, retrieve:
- Last 3 tier-0 summaries (3 days detail)
- Last 3 tier-1 summaries (9 days)
- Last 3 tier-2 summaries (27 days)
- ... up to highest tier

## Technical Design

### LLM

- Model: `gpt-4.1-mini`
- Max tokens per call: ~10,000 (configurable)
- Observations and model assignments via tool calls

### Database

SQLite with sqlite-vec for embeddings.

**Tables:**
- `observations` - id, text, timestamp, model_id, importance, embedding
- `summaries` - id, model_id, tier, text, start_timestamp, end_timestamp, embedding
- `models` - id, name, description, is_base

### Chunking

When processing exceeds MAX_TOKENS:
1. Estimate tokens (~4 chars per token)
2. Split into chunks under limit
3. Process each chunk separately
4. LLM emits multiple tool calls per chunk

## Tool Definitions

### add_observation

```python
{
    "name": "add_observation",
    "parameters": {
        "text": {"type": "string", "description": "First-person observation"},
        "importance": {"type": "integer", "description": "1-10+ scale"}
    }
}
```

### assign_model

```python
{
    "name": "assign_model",
    "parameters": {
        "observation_id": {"type": "integer"},
        "model_name": {"type": "string", "description": "Existing or new model name"}
    }
}
```

## CLI Commands

- `observe <text> [--importance N]` - add observation manually
- `list` - list observations
- `bootstrap --source <db>` - extract observations from conversation history
- `summarize [--tier N]` - run summarization
- `models` - list models with descriptions
- `model <name>` - show pyramid for a model
- `search <query>` - semantic search
