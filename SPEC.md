# Pyramid Memory System Specification

## Overview

A pyramidal memory system for AI agents. The agent reflects on information and generates observations, which are compressed into tiered summaries organized around mental models, then synthesized into coherent narratives.

## Core Concepts

### Observations

- Short text strings (usually single sentences) capturing specific facts
- Stored with timestamp
- Initially unassigned to any model (model_id=NULL)
- Created via LLM tool calls: `add_observation(text)`

### Models

Mental models about concepts - self, users, people, places, topics.

**Base models (always present):**
- `self` - the agent's own experience as an independent entity
- `user` - the primary user's identity, preferences, projects
- `system` - the technical environment and tools

**Discovered models:** Created during summarization when the LLM identifies distinct entities.

Each model has:
- `name` - unique identifier (lowercase, hyphenated)
- `description` - derived from observations
- `is_base` - true for self/user/system

### Summaries

Narrative prose summaries organized by tier.

**Constants:**
- `STEP = 10` - items per summary

**Tier structure:**
- Tier 0: Summarizes 10 observations
- Tier 1: Summarizes 10 tier-0 summaries
- Tier N: Summarizes 10 tier-(N-1) summaries

Summaries are written in clear, readable narrative prose. Importance is conveyed through word choice rather than markers.

### Pyramid Retrieval

For any model, retrieve all summaries from each tier, ordered by tier (highest first) and timestamp (newest first).

### Model Synthesis

When exporting, the pyramid and any unsummarized observations are synthesized into a coherent mental model using LLM. Newer details override older ones, facts are deduplicated, and the result is third-person narrative prose.

## Technical Design

### LLM

- Model: `gpt-4.1-mini`
- Max tokens per call: ~10,000 (configurable)
- Observations and model assignments via tool calls

### Database

SQLite with sqlite-vec for embeddings.

**Tables:**
- `observations` - id, text, timestamp, model_id
- `summaries` - id, model_id, tier, text, start_timestamp, end_timestamp
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
        "text": {"type": "string", "description": "Single factual sentence"}
    }
}
```

### assign_model

```python
{
    "name": "assign_model",
    "parameters": {
        "observation_id": {"type": "integer"},
        "model_name": {"type": "string", "description": "self, user, system, or new topic name"}
    }
}
```

## CLI Commands

- `observe <text>` - add observation manually
- `list [-n N]` - list observations
- `bootstrap --source <db>` - extract observations from conversation history
- `summarize [--clean] [--max-obs N] [--max-tier N]` - run summarization
- `summaries [--tier N]` - list summaries
- `models` - list models with descriptions
- `model <name>` - show pyramid for a model
- `embed` - generate embeddings
- `search <query> [--raw]` - semantic search
