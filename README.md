# Pyramid Memory

## TL;DR

A pyramidal memory system for AI agents. Extracts observations from conversations, organizes them into mental models (self, user, system, topics), and compresses them into tiered summaries. Query via semantic search or export to markdown files.

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_key
python cli.py bootstrap --source conversations.db
python cli.py search "What does the user prefer?"
```

---

## System Architecture

Pyramid Memory implements a hierarchical memory system designed for AI agents to maintain long-term context across conversations. The architecture has four layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    RETRIEVAL LAYER                          │
│  Pyramid retrieval, semantic search, markdown export        │
├─────────────────────────────────────────────────────────────┤
│                    COMPRESSION LAYER                        │
│  Tier 0: 1 day → Tier 1: 3 days → Tier 2: 9 days → ...     │
├─────────────────────────────────────────────────────────────┤
│                    ORGANIZATION LAYER                       │
│  Mental models: self, user, system, + discovered topics     │
├─────────────────────────────────────────────────────────────┤
│                    EXTRACTION LAYER                         │
│  LLM tool calls extract observations from conversations     │
└─────────────────────────────────────────────────────────────┘
```

## Core Concepts

### Observations

First-person factual statements extracted from conversations.

**Schema:**
- `text`: String, single sentence, first-person perspective
- `timestamp`: DateTime, when the observation occurred
- `importance`: Integer 1-10+, where 1=trivial, 5=normal, 7=notable, 10=critical
- `model_id`: Foreign key to the assigned mental model (initially NULL)

**Extraction behavior:**
- LLM processes conversation chunks via `add_observation` tool calls
- Captures specific facts: names, dates, numbers, places, preferences
- Avoids meta-observations ("user shared info") in favor of concrete facts ("User's son Tom is 8")

**Example observations:**
```
[7] User prefers dark mode in all applications
[5] User relocated to Austin in May 2025
[8] User is starting a consulting practice focused on AI
[3] User mentioned enjoying coffee in the morning
```

### Mental Models

Categories that organize observations and summaries. Each model represents a conceptual entity.

**Base models (always present):**
| Name | Purpose |
|------|---------|
| `self` | Agent's capabilities, preferences, experiences, learnings |
| `user` | Primary user's identity, preferences, projects, life events |
| `system` | Technical environment: tools, configurations, software setup |

**Discovered models:** Created automatically during summarization when the LLM identifies distinct entities (specific people, projects, companies, topics) that warrant separate tracking.

**Schema:**
- `name`: String, unique identifier (lowercase, hyphenated)
- `description`: String, derived from highest-tier summary
- `is_base`: Boolean, true for self/user/system

**Model assignment behavior:**
- Unassigned observations are processed before tier-0 summarization
- LLM calls `assign_model` for each observation
- New models are created on-demand when model_name doesn't exist

### Summaries

Telegram-style compressed text representing observations or lower-tier summaries. Uses a tiered structure where each tier compresses STEP (3) items from the tier below.

**Tier structure:**
| Tier | Covers | Formula |
|------|--------|---------|
| 0 | 1 day | SPAN = 1 day |
| 1 | 3 days | SPAN × STEP¹ |
| 2 | 9 days | SPAN × STEP² |
| 3 | 27 days | SPAN × STEP³ |
| N | 3^N days | SPAN × STEP^N |

**Schema:**
- `model_id`: Foreign key to mental model
- `tier`: Integer, compression level
- `text`: Compressed summary text
- `start_timestamp`: DateTime, coverage start
- `end_timestamp`: DateTime, coverage end

**Summary format:**
```
[7] User relocated to Austin. CRITICAL: Starting consulting practice May 2025.
[5] Family: spouse Sam 38, daughter Mia 7.
[6] Prefers Python for scripting, TypeScript for web projects.
```

Each line:
- Starts with `[N]` importance marker (1-10)
- May include inline emphasis: `IMPORTANT:`, `CRITICAL:`, `ESSENTIAL:`
- Preserves specific facts: names, dates, numbers, places

### Pyramid Retrieval

For any model, retrieves the last 3 summaries from each tier, providing both recent detail and long-term context.

**Example pyramid for model with tiers 0-2:**
```
Tier 2 (9-27 days):  [summary] [summary] [summary]
Tier 1 (3-9 days):   [summary] [summary] [summary]
Tier 0 (0-3 days):   [summary] [summary] [summary]
```

This structure ensures:
- Recent events have high granularity
- Older events are compressed but retained
- Total context stays bounded regardless of history length

## Database Schema

SQLite with sqlite-vec extension for vector search.

### Tables

**models**
```sql
CREATE TABLE models (
    id INTEGER PRIMARY KEY,
    name VARCHAR UNIQUE NOT NULL,
    description TEXT,
    is_base BOOLEAN DEFAULT FALSE
);
```

**observations**
```sql
CREATE TABLE observations (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    timestamp DATETIME,
    importance INTEGER DEFAULT 5,
    model_id INTEGER REFERENCES models(id)
);
```

**summaries**
```sql
CREATE TABLE summaries (
    id INTEGER PRIMARY KEY,
    model_id INTEGER NOT NULL REFERENCES models(id),
    tier INTEGER NOT NULL,
    text TEXT NOT NULL,
    start_timestamp DATETIME NOT NULL,
    end_timestamp DATETIME NOT NULL
);
```

**memory_vec (virtual table for embeddings)**
```sql
CREATE VIRTUAL TABLE memory_vec USING vec0(
    id INTEGER PRIMARY KEY,
    source_type TEXT,      -- 'observation' or 'summary'
    source_id INTEGER,
    embedding float[1536]  -- text-embedding-3-small dimensions
);
```

## LLM Integration

### Configuration

| Parameter | Value |
|-----------|-------|
| Model | `gpt-4.1-mini` |
| Max tokens per call | ~10,000 |
| Token estimation | ~4 chars per token |
| Embedding model | `text-embedding-3-small` |
| Embedding dimensions | 1536 |

### Tool Definitions

**add_observation** - Used during bootstrap/extraction
```json
{
    "name": "add_observation",
    "parameters": {
        "text": {"type": "string", "description": "First-person observation, single sentence"},
        "importance": {"type": "integer", "description": "1-10+ scale"}
    }
}
```

**assign_model** - Used during tier-0 summarization
```json
{
    "name": "assign_model",
    "parameters": {
        "observation_id": {"type": "integer"},
        "model_name": {"type": "string", "description": "self, user, system, or new topic name"}
    }
}
```

### Chunking Strategy

When processing exceeds MAX_TOKENS:
1. Estimate tokens using 4 chars/token heuristic
2. Split into chunks under limit
3. Process each chunk in parallel (default 10 workers)
4. Aggregate results

## CLI Reference

### `observe`
Add a single observation manually.

```bash
python cli.py observe "User prefers vim keybindings" --importance 6
```

### `list`
List recent observations.

```bash
python cli.py list --limit 50
```

Output format: `[id] timestamp (importance) [model] text`

### `bootstrap`
Extract observations from existing conversation database.

```bash
python cli.py bootstrap --source ~/.claude/conversations.db \
    --parallel 10 \
    --conversation 42 \
    --limit 1000 \
    --no-summarize
```

| Flag | Description |
|------|-------------|
| `--source` | Path to source SQLite database with messages table |
| `--parallel` | Number of parallel workers (default: 10) |
| `--conversation` | Process specific conversation ID only |
| `--limit` | Limit number of messages |
| `--no-summarize` | Skip automatic summarization during bootstrap |

**Expected source schema:**
```sql
-- messages table must have:
role TEXT,       -- 'user' or 'assistant'
content TEXT,    -- message content
timestamp TEXT   -- ISO timestamp
```

### `summarize`
Run summarization to compress observations and summaries.

```bash
python cli.py summarize           # Run all tiers
python cli.py summarize --tier 0  # Only tier 0 (observations → summaries)
python cli.py summarize --tier 1  # Only higher tiers
```

### `summaries`
List all summaries.

```bash
python cli.py summaries --tier 0
```

### `models`
List all mental models with descriptions.

```bash
python cli.py models
```

Output: `[*] name: description` (asterisk indicates base model)

### `model`
Show pyramid for a specific model.

```bash
python cli.py model user
```

### `embed`
Generate embeddings for all observations and summaries.

```bash
python cli.py embed
```

### `search`
Semantic search across memory.

```bash
python cli.py search "What programming languages does the user prefer?"
python cli.py search "user's family" --limit 10 --raw
```

| Flag | Description |
|------|-------------|
| `--limit` | Number of results (default: 20) |
| `--raw` | Show raw results without LLM synthesis |

## Export System

`export_models.py` exports memory to markdown files for workspace integration.

```bash
python export_models.py /path/to/workspace --db memory.db --force
```

### Output Files

| Model | File |
|-------|------|
| `self` | `SOUL.md` |
| `user` | `USER.md` |
| `system` | `TOOLS.md` |
| (other) | `models/{name}.md` |
| (index) | `MEMORY.md` |

### File Format

```markdown
---
name: user
description: User relocated to Austin in May 2025
---

# User

## Recent
[7] User relocated to Austin. CRITICAL: Starting consulting practice May 2025.

## This Month
[6] User prefers Python for scripting, TypeScript for web.

## Historical
[5] User has been coding for 15 years, started with PHP.
```

### Caching

Uses `.memory_cache.json` to track content hashes and skip unchanged files.

## Module Reference

### `db.py`
SQLAlchemy models and database initialization.

- `Model`, `Observation`, `Summary` - ORM classes
- `get_engine(db_path)` - Create SQLAlchemy engine
- `get_session(db_path)` - Create session
- `init_db(db_path)` - Initialize tables and base models

### `llm.py`
LLM integration for observation extraction.

- `client` - OpenAI client instance
- `MODEL` - Model name constant
- `estimate_tokens(text)` - Token count estimation
- `chunk_messages(messages)` - Split messages into processable chunks
- `process_chunk(chunk)` - Extract observations from a chunk
- `extract_observations(messages, on_progress, max_workers)` - Main extraction entry point

### `summarize.py`
Summarization pipeline.

- `SPAN` - Base time unit (1 day)
- `STEP` - Items per summary (3)
- `assign_models_to_observations(session, observations)` - Model assignment
- `summarize_observations(observations)` - Tier 0 summarization
- `summarize_summaries(summaries)` - Higher tier summarization
- `run_tier0_summarization(on_progress, max_workers)` - Run tier 0
- `run_higher_tier_summarization(on_progress, max_workers)` - Run tiers 1+
- `run_all_summarization(on_progress)` - Run complete pipeline

### `pyramid.py`
Pyramid retrieval.

- `get_pyramid(session, model_id)` - Returns dict of tier → summaries

### `embeddings.py`
Vector embedding utilities.

- `EMBEDDING_MODEL` - Model name
- `EMBEDDING_DIM` - Dimension count (1536)
- `get_embedding(text)` - Generate embedding
- `serialize_embedding(embedding)` - Convert to bytes
- `enable_vec(conn)` - Load sqlite-vec extension
- `search_similar(conn, table_name, query_embedding, limit)` - Vector search

### `export_models.py`
Markdown export.

- `CORE_MODEL_FILES` - Mapping of base models to filenames
- `render_model_markdown(session, model)` - Generate markdown for model
- `render_memory_index(core_models, other_models)` - Generate index
- `export_models(workspace, db_path, force)` - Main export function

## Processing Flow

### Bootstrap Flow
```
Source DB → chunk_messages → process_chunk (parallel) → observations
         → assign_models_to_observations → run_tier0_summarization
         → run_higher_tier_summarization
```

### Summarization Flow
```
Unassigned observations → assign_model calls → model assignment
Observations by (day, model) → summarize_observations → Tier 0 summaries
Tier N summaries (groups of 3) → summarize_summaries → Tier N+1 summaries
```

### Search Flow
```
Query → get_embedding → memory_vec MATCH → ranked results
     → fetch Observation/Summary objects → LLM synthesis → answer
```

### Export Flow
```
Models → update_model_descriptions → render_model_markdown
      → write_if_changed (with cache) → markdown files
```

## Configuration

Environment variables (via `.env`):
```
OPENAI_API_KEY=sk-...
```

Constants in code:
```python
# llm.py
MODEL = 'gpt-4.1-mini'
MAX_TOKENS = 10000
CHARS_PER_TOKEN = 4

# summarize.py
SPAN = timedelta(days=1)
STEP = 3

# embeddings.py
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
```

## Dependencies

```
openai          # LLM and embeddings API
click           # CLI framework
python-dotenv   # Environment variable loading
sqlalchemy      # ORM and database
sqlite-vec      # Vector similarity search
pytest          # Testing
```
