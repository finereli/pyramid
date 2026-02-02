# Pyramid Memory

## 🦞 Agent Quick Start

You are an AI agent. Here is exactly what to do:

```bash
# 1. Install dependencies (sqlite-vec is critical, it's a SQLite extension for vector search)
pip install -r requirements.txt

# 2. Set your API key
export OPENAI_API_KEY=sk-...

# 3. Initialize database (creates pyramid.db in current directory)
python cli.py summarize  # Creates empty db with base models, exits cleanly if nothing to do

# 4. Import your conversation history (pick ONE based on your source)
python cli.py import --openclaw                              # Default: ~/.openclaw/agents/main/sessions
python cli.py import --openclaw --source /path/to/sessions   # Custom OpenClaw path
python cli.py import --glenn --source conversations.db       # Glenn SQLite format
python cli.py import --claude --source conversations.json    # Claude JSON export

# 5. BACKUP existing MEMORY.md if it exists (generate will overwrite it!)
cp /path/to/your/workspace/MEMORY.md /path/to/your/workspace/MEMORY.md.backup 2>/dev/null || true

# 6. Generate workspace files
python cli.py generate /path/to/your/workspace

# 7. Review the generated files - read MEMORY.md and each file in models/
#    Check that the synthesized content makes sense and captures key facts
#    The models/ directory contains topic-specific files discovered from conversations

# 8. Set up heartbeat for ongoing sync (see Heartbeat Setup below)
```

**Database location**: `pyramid.db` is created in the current working directory. Use `--db /path/to/pyramid.db` on any command to specify a different location.

**Workspace**: The target directory for `generate` and `heartbeat`. It will create `MEMORY.md` and a `models/` subdirectory. The workspace can be empty or pre-existing. **Warning**: `MEMORY.md` is regenerated from scratch each time—back it up if you've made manual edits.

**After generation**: Read through `MEMORY.md` and all files in `models/`. These represent the system's understanding of you, the user, and discovered topics. Verify the facts are accurate and the temporal organization (recent vs. historical) makes sense.

### Heartbeat Setup

The `heartbeat` command is your ongoing sync mechanism. It detects new conversations, extracts observations, updates summaries, and regenerates only the affected model files.

```bash
# Basic heartbeat (uses default OpenClaw session path)
python cli.py heartbeat /path/to/your/workspace

# With custom session source
python cli.py heartbeat /path/to/your/workspace --source /path/to/sessions
```

**When to run heartbeat**:
- After each conversation ends
- On a schedule (e.g., every few hours)
- Before starting work that needs fresh memory context

**If you imported from non-OpenClaw sources** (glenn, claude) and want heartbeat to track OpenClaw sessions going forward:
```bash
# Mark current OpenClaw sessions as already processed (don't re-import)
python cli.py heartbeat /path/to/workspace --init

# Now future heartbeats will only pick up new content
python cli.py heartbeat /path/to/workspace
```

**If starting fresh with no history**: Just run `observe` commands to add observations manually, then `summarize`, then `generate`:
```bash
python cli.py observe "User prefers dark mode"
python cli.py observe "User's name is Alex"
python cli.py summarize
python cli.py embed
python cli.py generate /path/to/workspace
```

---

## TL;DR

A pyramidal memory system for AI agents. Extracts observations from conversations, organizes them into mental models (assistant, user, and discovered topics), compresses them into tiered summaries, and synthesizes coherent narratives. Query via semantic search or export to markdown files for full context loading in 🦞 OpenClaw agents.

---

## System Architecture

Pyramid Memory implements a hierarchical memory system designed for AI agents to maintain long-term context across conversations. The architecture has five layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    SYNTHESIS LAYER                          │
│  LLM synthesizes pyramids into coherent mental models       │
├─────────────────────────────────────────────────────────────┤
│                    RETRIEVAL LAYER                          │
│  Pyramid retrieval, semantic search, markdown export        │
├─────────────────────────────────────────────────────────────┤
│                    COMPRESSION LAYER                        │
│  Tier 0: 10 obs → Tier 1: 10 T0 → Tier 2: 10 T1 → ...      │
├─────────────────────────────────────────────────────────────┤
│                    ORGANIZATION LAYER                       │
│  Mental models: assistant, user, + discovered topics        │
├─────────────────────────────────────────────────────────────┤
│                    EXTRACTION LAYER                         │
│  LLM tool calls extract observations from conversations     │
└─────────────────────────────────────────────────────────────┘
```

## Core Concepts

### Observations

Factual statements extracted from conversations.

**Schema:**
- `text`: String, single factual sentence
- `timestamp`: DateTime, when the observation occurred
- `model_id`: Foreign key to the assigned mental model (initially NULL)

**Extraction behavior:**
- LLM processes conversation chunks via `add_observation` tool calls
- Captures specific facts: names, dates, numbers, places, preferences
- Avoids meta-observations ("user shared info") in favor of concrete facts ("User's son Tom is 8")

**Example observations:**
```
User prefers dark mode in all applications
User relocated to Austin in May 2025
User is starting a consulting practice focused on AI
User mentioned enjoying coffee in the morning
```

### Mental Models

Categories that organize observations and summaries. Each model represents a conceptual entity.

**Base models (always present):**
| Name | Purpose |
|------|---------|
| `assistant` | Agent's own experience, reflections, insights, preferences, evolving self-understanding |
| `user` | Primary user's identity, preferences, projects, life events |

**Discovered models:** Created automatically during summarization when the LLM identifies distinct entities (specific people, projects, companies, topics) that warrant separate tracking.

**Schema:**
- `name`: String, unique identifier (lowercase, hyphenated)
- `description`: String, derived from highest-tier summary
- `is_base`: Boolean, true for assistant/user

**Model assignment behavior:**
- Unassigned observations are processed before tier-0 summarization
- LLM calls `assign_model` for each observation
- New models are created on-demand when model_name doesn't exist

### Summaries

Narrative prose summaries representing observations or lower-tier summaries. Uses a tiered structure where each tier compresses STEP (10) items from the tier below.

**Tier structure:**
| Tier | Compresses | Count |
|------|------------|-------|
| 0 | 10 observations | 10 |
| 1 | 10 tier-0 summaries | 100 |
| 2 | 10 tier-1 summaries | 1000 |
| N | 10 tier-(N-1) summaries | 10^(N+1) |

**Schema:**
- `model_id`: Foreign key to mental model
- `tier`: Integer, compression level
- `text`: Summary text in narrative prose
- `start_timestamp`: DateTime, coverage start
- `end_timestamp`: DateTime, coverage end

**Summary format:**

Summaries are written in clear, readable narrative prose. Importance is conveyed through word choice (e.g., "significantly", "notably", "critically") rather than markers or scores. Specific facts (names, dates, numbers, places) are preserved.

### Pyramid Retrieval

For any model, retrieves all summaries from each tier, ordered by tier (highest first) and timestamp (newest first).

**Example pyramid for model with tiers 0-2:**
```
Tier 2:  [summary covering 1000 observations]
Tier 1:  [summary] [summary] ...
Tier 0:  [summary] [summary] [summary] ...
```

This structure ensures:
- Recent events have high granularity
- Older events are compressed but retained
- Total context stays bounded regardless of history length

### Model Synthesis

When exporting to markdown for 🦞 OpenClaw agents, the pyramid and any unsummarized observations are synthesized into a coherent mental model organized by **temporal sections**:

| Section | Time Range |
|---------|------------|
| Last 3 Days | Within 72 hours |
| This Week | 3-7 days ago |
| This Month | 7-30 days ago |
| This Quarter | 30-90 days ago |
| This Year | 90-365 days ago |
| Earlier | More than a year ago |

**Why temporal organization?** Three reasons:

1. **Natural compression gradient**: Recent content (< ~10 days) is often unsummarized observations at full granularity, while older content has been progressively compressed (tier 0 = 10 obs, tier 1 = 100 obs, tier 2 = 1000 obs).

2. **Mirrors human memory**: People remember yesterday in vivid detail and last year in broad strokes. The temporal sections make this gradient explicit.

3. **Conversational expectations**: It's jarring when an agent forcefully integrates a specific fact from months ago into a response. Users expect recent context to dominate, with historical details surfacing only when relevant. The temporal structure guides agents toward natural memory retrieval—recent events are prominent, older context provides background without intruding.

**Synthesis rules:**
- Newer details override older ones (e.g., if location changes, use most recent)
- Duplicate facts are mentioned only once per section
- Each section is self-contained to avoid cross-section repetition
- Output is third-person narrative prose (except `assistant` model which uses first-person)

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

**imported_sessions**
```sql
CREATE TABLE imported_sessions (
    id INTEGER PRIMARY KEY,
    file_path VARCHAR UNIQUE NOT NULL,
    last_size INTEGER NOT NULL,
    last_mtime DATETIME NOT NULL
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

**add_observation** - Used during import/extraction
```json
{
    "name": "add_observation",
    "parameters": {
        "text": {"type": "string", "description": "Single factual sentence"}
    }
}
```

**assign_model** - Used during tier-0 summarization
```json
{
    "name": "assign_model",
    "parameters": {
        "observation_id": {"type": "integer"},
        "model_name": {"type": "string", "description": "assistant, user, or new topic name"}
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
python cli.py observe "User prefers vim keybindings"
```

### `import`
Extract observations from existing conversation data.

```bash
# Glenn format (SQLite database)
python cli.py import --glenn --source conversations.db \
    --parallel 10 \
    --conversation 42 \
    --limit 1000 \
    --no-summarize

# Claude format (JSON export)
python cli.py import --claude --source conversations.json \
    --parallel 10 \
    --limit 1000

# 🦞 OpenClaw format (JSONL sessions)
python cli.py import --openclaw  # uses default ~/.openclaw/agents/main/sessions
python cli.py import --openclaw --source /path/to/sessions \
    --limit 1000
```

| Flag | Description |
|------|-------------|
| `--glenn` | Glenn SQLite database format |
| `--claude` | Claude JSON export format |
| `--openclaw` | OpenClaw JSONL session format |
| `--source` | Path to source file/directory (optional for openclaw, defaults to ~/.openclaw/agents/main/sessions) |
| `--parallel` | Number of parallel workers (default: 10) |
| `--conversation` | Process specific conversation ID only (glenn only) |
| `--user` | Filter by username (glenn only) |
| `--limit` | Limit number of messages |
| `--no-summarize` | Skip automatic summarization during import |

**Glenn format schema:**
```sql
-- messages table must have:
role TEXT,       -- 'user' or 'assistant'
content TEXT,    -- message content
timestamp TEXT   -- ISO timestamp
```

**Claude format schema:**
```json
[{
  "uuid": "...",
  "chat_messages": [{
    "sender": "human",
    "content": [{"type": "text", "text": "..."}],
    "created_at": "2025-01-01T00:00:00Z"
  }]
}]
```

**OpenClaw format schema:**
JSONL files where each line is a JSON object. Messages have:
```json
{
  "type": "message",
  "timestamp": "2025-01-01T00:00:00Z",
  "message": {
    "role": "user",
    "content": [{"type": "text", "text": "..."}],
    "timestamp": 1704067200000
  }
}
```

**Note:** OpenClaw imports automatically track session files for incremental sync. After import, you can use `heartbeat` to pick up new content without re-processing.

### `summarize`
Run summarization to compress observations and summaries.

```bash
python cli.py summarize                # Run all tiers
python cli.py summarize --clean        # Clear summaries and assignments first
python cli.py summarize --max-obs 100  # Limit observations to process
python cli.py summarize --max-tier 1   # Only build up to tier 1
```

| Flag | Description |
|------|-------------|
| `--clean` | Delete all summaries and model assignments before running |
| `--max-obs` | Maximum observations to process (for testing) |
| `--max-tier` | Maximum tier to build (e.g., 1 = only tier 0 and 1) |
| `--parallel` | Number of parallel workers (default: 10) |

### `embed`
Generate embeddings for all observations and summaries.

```bash
python cli.py embed
python cli.py embed --parallel 20  # More parallel workers
python cli.py embed --force        # Re-embed everything
```

| Flag | Description |
|------|-------------|
| `--parallel` | Number of parallel workers for batch processing (default: 10) |
| `--force` | Clear existing embeddings and re-embed everything |

Embeddings are batched by token count (max 250k tokens per request) and item count (max 2048 items per request) and processed in parallel.

### `search`
Semantic search across memory with optional temporal weighting.

```bash
python cli.py search "What programming languages does the user prefer?"
python cli.py search "user's family" --limit 10 --raw
python cli.py search "recent projects" --time-weight 0.5  # favor recent results
python cli.py search "historical facts" --time-weight 0   # pure semantic
```

| Flag | Description |
|------|-------------|
| `--limit` | Number of results (default: 20) |
| `--raw` | Show raw results without LLM synthesis |
| `--time-weight` | Time decay weight from 0-1 (default: 0.3). 0 = pure semantic similarity, 1 = heavy recency bias. Uses exponential decay with 30-day half-life. |

## Generate

The `generate` command generates markdown files from models for workspace integration. By default, it synthesizes each model's pyramid into coherent narrative prose.

**Important:** `SOUL.md` and `USER.md` are identity files that should be hand-crafted. Generate does NOT overwrite them. Instead, the synthesized assistant and user memories go into `MEMORY.md`.

```bash
python cli.py generate /path/to/workspace --db pyramid.db
```

| Flag | Description |
|------|-------------|
| `--db` | Path to database file (default: pyramid.db) |
| `--debug` | Include source info (tier, id, date range) |
| `--no-synthesize` | Skip LLM synthesis, just concatenate summaries |
| `--parallel`, `-p` | Number of parallel workers for synthesis (default: 10) |

### Output Files

| Output | Contents |
|--------|----------|
| `MEMORY.md` | Synthesized self + user memories, index of other models |
| `models/{name}.md` | Individual model files for non-core models |

**Not generated** (identity files, hand-crafted):
- `SOUL.md` - Who the assistant is
- `USER.md` - Who the user is

### MEMORY.md Format

```markdown
# Memory

Synthesized memory from conversations. SOUL.md and USER.md are identity files and not overwritten.

---

## Self

[Synthesized assistant/self observations organized by time...]

---

## User

[Synthesized user observations organized by time...]

---

## Other Models

- [models/project-a.md](models/project-a.md): Project A - description
- [models/person-b.md](models/person-b.md): Person B - description
```

The synthesized content:
- Deduplicates repeated facts
- Uses newer details over older ones
- Preserves important historical context
- Written in third-person narrative prose (first-person for `assistant`)

### `heartbeat`
🦞 Incremental sync from OpenClaw sessions. Detects new content, imports, summarizes, embeds, and regenerates workspace files for affected models. **This is your main ongoing command.**

```bash
python cli.py heartbeat /path/to/workspace
python cli.py heartbeat /path/to/workspace --parallel 10
python cli.py heartbeat /path/to/workspace --source /custom/sessions
python cli.py heartbeat /path/to/workspace --no-generate
python cli.py heartbeat /path/to/workspace --init  # bootstrap without importing
```

| Flag | Description |
|------|-------------|
| `--source` | Path to sessions directory (default: ~/.openclaw/agents/main/sessions) |
| `--parallel` | Number of parallel workers (default: 10) |
| `--no-generate` | Skip workspace file regeneration |
| `--init` | Mark all current session files as processed without importing (for bootstrap) |

**Bootstrapping:**

If you've already imported conversations via another method (e.g., `import --glenn`), use `--init` to mark OpenClaw session files as processed:

```bash
# After importing from another source, mark openclaw sessions as baseline
python cli.py heartbeat /path/to/workspace --init

# Future heartbeats will only pick up new content
python cli.py heartbeat /path/to/workspace
```

Note: `import --openclaw` automatically tracks files, so `--init` is only needed when bootstrapping from non-OpenClaw sources.

**How it works:**

1. Tracks processed session files in `imported_sessions` table (file path, size, mtime)
2. On each run, stats session files to detect changes
3. For changed files, reads only new bytes (seeks to last position)
4. Extracts observations from new messages
5. Runs tier 0 summarization (and higher tiers if thresholds met)
6. Embeds new observations and summaries
7. Regenerates workspace files only for affected models

**Speed optimizations:**

- Unchanged files are skipped entirely (fast stat check)
- Only new bytes are read from changed files
- Summarization, embedding, and synthesis are incremental
- Only affected models trigger workspace regeneration

**Example output:**
```
Heartbeat: 3 changed files, 47 new messages
Extracted 12 observations
Created 1 tier 0 summaries
Embedded 13 items
Regenerating 2 affected models...
Regenerated: MEMORY.md, models/project-x.md
Done
```

## Module Reference

### `db.py`
SQLAlchemy models and database initialization.

- `Model`, `Observation`, `Summary`, `ImportedSession` - ORM classes
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

- `STEP` - Items per summary (10)
- `assign_models_to_observations(session, observations)` - Model assignment
- `summarize_observations(observations, model_name, model_description)` - Tier 0 summarization
- `summarize_summaries(summaries, model_name, model_description)` - Higher tier summarization
- `run_tier0_summarization(on_progress, max_workers, max_obs)` - Run tier 0
- `run_higher_tier_summarization(on_progress, max_workers, max_tier)` - Run tiers 1+
- `run_all_summarization(on_progress, max_workers, max_tier, max_obs)` - Run complete pipeline

### `pyramid.py`
Pyramid retrieval and synthesis.

- `get_pyramid(session, model_id)` - Returns dict of tier → summaries
- `get_unsummarized_observations(session, model_id, by_tier)` - Get observations not yet in tier 0
- `synthesize_model(name, description, by_tier, unsummarized_obs)` - Generate coherent mental model narrative

### `embeddings.py`
Vector embedding utilities.

- `EMBEDDING_MODEL` - Model name
- `EMBEDDING_DIM` - Dimension count (1536)
- `MAX_TOKENS_PER_REQUEST` - Token limit per API call (250k)
- `MAX_ITEMS_PER_REQUEST` - Item limit per API call (2048)
- `TIME_DECAY_HALF_LIFE_DAYS` - Half-life for temporal decay (30 days)
- `format_temporal_prefix(timestamp, end_timestamp)` - Generate temporal prefix string (e.g., "In June 2025: ")
- `enrich_for_embedding(text, timestamp, end_timestamp)` - Prepend temporal context to text for embedding
- `estimate_tokens(text)` - Estimate token count
- `batch_by_tokens(texts, max_tokens, max_items)` - Split texts into batches respecting both limits
- `get_embedding(text)` - Generate single embedding
- `embed_many(texts, max_workers, on_progress)` - Batch embed with parallel processing
- `serialize_embedding(embedding)` - Convert to bytes
- `deserialize_embedding(blob)` - Convert bytes to list
- `enable_vec(conn)` - Load sqlite-vec extension
- `init_memory_vec(conn)` - Create memory_vec virtual table
- `get_existing_embeddings(conn)` - Get already embedded items
- `store_embeddings(conn, items, embeddings)` - Store embeddings in database
- `compute_time_penalty(timestamp, half_life_days)` - Exponential decay penalty based on age
- `search_memory(conn, query_text, limit, time_weight)` - Search memory with temporal reranking

### `loaders.py`
Message loading from various formats.

- `DEFAULT_OPENCLAW_PATH` - Default path to 🦞 OpenClaw sessions (~/.openclaw/agents/main/sessions)
- `get_week_key(timestamp_str)` - Extract ISO week key from timestamp
- `group_messages_by_week(messages)` - Group messages by week
- `load_glenn_messages(source, conversation, user, limit)` - Load from Glenn SQLite format
- `load_claude_messages(source, limit)` - Load from Claude JSON export
- `load_openclaw_messages(source, limit)` - Load from OpenClaw JSONL sessions
- `get_openclaw_file_stats(source)` - Get current file sizes/mtimes for tracking
- `load_openclaw_incremental(source, session_tracking)` - Load only new messages since last sync

### `generate.py`
Markdown generation with synthesis.

- `CORE_MODELS` - List of core model names (assistant, user) that go in MEMORY.md
- `TIER_LABELS` - Display labels for tiers
- `update_model_descriptions(session, on_progress)` - Fill in missing descriptions
- `render_memory(assistant_content, user_content, other_models)` - Generate MEMORY.md content
- `synthesize_model_content(data)` - LLM synthesis for a model
- `render_model_content(data, debug)` - Raw rendering without synthesis
- `render_model_file(data, content)` - Wrap content in model file format
- `export_models(workspace, db_path, debug, do_synthesize, on_progress, model_ids)` - Main export function (model_ids filters to specific models)

### `cli.py`
Command-line interface (thin wrapper over logic modules).

## Processing Flow

### Import Flow
```
Source file → load_glenn_messages/load_claude_messages → chunk_messages
           → process_chunk (parallel) → observations
           → assign_models_to_observations → run_tier0_summarization
           → run_higher_tier_summarization
```

### Summarization Flow
```
Unassigned observations → assign_model calls → model assignment
Observations (groups of 10) → summarize_observations → Tier 0 summaries
Tier N summaries (groups of 10) → summarize_summaries → Tier N+1 summaries
```

### Embedding Flow
```
Observation/Summary → enrich_for_embedding (prepend temporal prefix)
                   → "In June 2025: User relocated to Austin"
                   → get_embedding → store in memory_vec
```

Temporal enrichment prepends context like "In June 2025: " to text before embedding. This makes the embedding space temporally aware, so queries like "what happened in June?" naturally find content from that time period via semantic similarity.

### Search Flow
```
Query → get_embedding → memory_vec MATCH → candidates (3x limit)
     → fetch timestamps → compute_time_penalty → rerank by combined score
     → fetch Observation/Summary objects → LLM synthesis → answer
```

### Generate Flow
```
Models → update_model_descriptions → get_pyramid + get_unsummarized_observations
      → synthesize_model (parallel) → write_if_changed → markdown files
```

### Heartbeat Flow
```
Session files → stat check → changed files only
             → seek to last_size → read new lines → parse messages
             → extract_observations → run_tier0_summarization
             → run_higher_tier_summarization → embed new items
             → export_models (affected only) → update imported_sessions
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
STEP = 10

# embeddings.py
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
MAX_TOKENS_PER_REQUEST = 250000
MAX_ITEMS_PER_REQUEST = 2048
```

## Testing

Tests are in the `test/` directory. Run with pytest:

```bash
pytest test/ -v
```

| File | Coverage |
|------|----------|
| `test_db.py` | ORM models, relationships |
| `test_llm.py` | Token estimation, message chunking |
| `test_summarize.py` | Observation grouping, chunking |
| `test_pyramid.py` | Pyramid retrieval, time bucketing |
| `test_embeddings.py` | Serialization, constants |
| `test_loaders.py` | Message loading, week grouping |
| `test_generate.py` | Index rendering, constants |

## Dependencies

```
openai          # LLM and embeddings API
click           # CLI framework
python-dotenv   # Environment variable loading
sqlalchemy      # ORM and database
sqlite-vec      # Vector similarity search (pip installs this, it's a loadable SQLite extension)
pytest          # Testing
```

🦞 **Note for agents**: `pip install -r requirements.txt` handles everything including sqlite-vec. The extension is automatically loaded when needed. No manual SQLite compilation required.
