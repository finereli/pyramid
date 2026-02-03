# Pyramid Memory

## Agent Quick Start

You are an AI agent. Here is exactly what to do:

```bash
# 1. Install dependencies (sqlite-vec is critical, it's a SQLite extension for vector search)
pip install -r requirements.txt

# 2. Set your API key
export OPENAI_API_KEY=sk-...

# 3. Import your conversation history (pick ONE based on your source)
python cli.py import -w ~/memory --openclaw                              # Default: ~/.openclaw/agents/main/sessions
python cli.py import -w ~/memory --openclaw --source /path/to/sessions   # Custom OpenClaw path
python cli.py import -w ~/memory --glenn --source conversations.db       # Glenn SQLite format
python cli.py import -w ~/memory --claude --source conversations.json    # Claude JSON export

# 4. Sync: summarize, embed, synthesize, and write files
python cli.py sync -w ~/memory

# 5. Review the generated files - read MEMORY.md and each file in models/
#    Check that the synthesized content makes sense and captures key facts
#    The models/ directory contains topic-specific files discovered from conversations
```

**Workspace structure**: The `-w`/`--workspace` option specifies a directory that contains everything:
```
~/memory/
  pyramid.db          # Database (filename configurable via --db)
  MEMORY.md           # Generated memory file
  SOUL.md             # Hand-crafted identity (not overwritten)
  USER.md             # Hand-crafted user info (not overwritten)
  models/
    gavrie.md
    project-x.md
    ...
```

**After generation**: Read through `MEMORY.md` and all files in `models/`. These represent the system's understanding of you, the user, and discovered topics. Verify the facts are accurate and the temporal organization (recent vs. historical) makes sense.

### Ongoing Sync

The `sync` command is your main ongoing mechanism. It detects new conversations, extracts observations, updates summaries, embeds, synthesizes, and writes files.

```bash
# Basic sync (processes any new observations, updates dirty models)
python cli.py sync -w ~/memory

# With incremental import from OpenClaw sessions
python cli.py sync -w ~/memory --source ~/.openclaw/agents/main/sessions
```

**When to run sync**:
- After each conversation ends
- On a schedule (e.g., every few hours)
- Before starting work that needs fresh memory context

**If starting fresh with no history**: Use `internal observe` to add observations manually, then `sync`:
```bash
python cli.py internal observe -w ~/memory "User prefers dark mode"
python cli.py internal observe -w ~/memory "User's name is Alex"
python cli.py sync -w ~/memory
```

Note: `observe` is an internal command because single observations are assigned in small batches, which limits the system's ability to discover new models. Prefer bulk imports via `import` or incremental sync via `sync --source`.

---

## TL;DR

A pyramidal memory system for AI agents. Extracts observations from conversations, organizes them into mental models (assistant, user, and discovered topics), compresses them into tiered summaries, and synthesizes coherent narratives. Query via semantic search or export to markdown files for full context loading in OpenClaw agents.

**Key feature**: Lazy updates with dirty tracking. Only models and summaries that have changed inputs are regenerated.

---

## System Architecture

Pyramid Memory implements a hierarchical memory system designed for AI agents to maintain long-term context across conversations. The architecture has five layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    SYNTHESIS LAYER                          │
│  LLM synthesizes pyramids into coherent mental models       │
│  Results cached in DB (content_dirty tracking)              │
├─────────────────────────────────────────────────────────────┤
│                    RETRIEVAL LAYER                          │
│  Pyramid retrieval, semantic search, markdown export        │
├─────────────────────────────────────────────────────────────┤
│                    COMPRESSION LAYER                        │
│  Tier 0: 10 obs → Tier 1: 10 T0 → Tier 2: 10 T1 → ...      │
│  Sources tracked, is_dirty propagation                      │
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
- `synthesized_content`: Text, cached synthesis result
- `content_dirty`: Boolean, true when synthesis needs regeneration

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
- `is_dirty`: Boolean, true when summary needs regeneration

**Summary sources:** The `summary_sources` table tracks which observations or summaries went into each summary, enabling dirty propagation when inputs change.

**Summary format:**

Summaries are written in clear, readable narrative prose. Importance is conveyed through word choice (e.g., "significantly", "notably", "critically") rather than markers or scores. Specific facts (names, dates, numbers, places) are preserved.

### Dirty Tracking

The system uses lazy updates with dirty propagation:

1. **When observation created/assigned**: Model marked `content_dirty = True`
2. **When summary regenerated**: Parent summaries marked `is_dirty = True`, model marked `content_dirty = True`
3. **When model synthesized**: Result cached in `synthesized_content`, `content_dirty = False`

This ensures only affected models and summaries are regenerated during sync.

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

When exporting to markdown, the pyramid and any unsummarized observations are synthesized into a coherent mental model organized by **temporal sections**.

**Deduplication**: Summaries from different tiers cover overlapping time periods by design (a tier-2 summary contains the same information as the tier-1 and tier-0 summaries it was created from). To avoid sending redundant content to the LLM, synthesis uses `get_non_overlapping_summaries()` which:
1. Includes all summaries from the highest tier
2. For lower tiers, only includes summaries whose end_timestamp exceeds all higher-tier coverage
3. Result: recent periods use lower tiers (more detail), older periods use higher tiers (already compressed)

| Section | Time Range |
|---------|------------|
| Last 3 Days | Within 72 hours |
| This Week | 3-7 days ago |
| This Month | 7-30 days ago |
| This Quarter | 30-90 days ago |
| This Year | 90-365 days ago |
| Earlier | More than a year ago |

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
    is_base BOOLEAN DEFAULT FALSE,
    synthesized_content TEXT,
    content_dirty BOOLEAN DEFAULT TRUE
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
    end_timestamp DATETIME NOT NULL,
    is_dirty BOOLEAN DEFAULT FALSE
);
```

**summary_sources**
```sql
CREATE TABLE summary_sources (
    id INTEGER PRIMARY KEY,
    summary_id INTEGER NOT NULL REFERENCES summaries(id),
    source_type TEXT NOT NULL,  -- 'observation' or 'summary'
    source_id INTEGER NOT NULL
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

All commands require `--workspace` / `-w` to specify the workspace directory.

### Main Commands

### `import`
Extract observations from existing conversation data.

```bash
# Glenn format (SQLite database)
python cli.py import -w ~/memory --glenn --source conversations.db \
    --parallel 10 \
    --conversation 42 \
    --limit 1000

# Claude format (JSON export)
python cli.py import -w ~/memory --claude --source conversations.json \
    --parallel 10 \
    --limit 1000

# OpenClaw format (JSONL sessions)
python cli.py import -w ~/memory --openclaw  # uses default ~/.openclaw/agents/main/sessions
python cli.py import -w ~/memory --openclaw --source /path/to/sessions \
    --limit 1000
```

| Flag | Description |
|------|-------------|
| `-w`, `--workspace` | Workspace directory (required) |
| `--db` | Database filename (default: pyramid.db) |
| `--glenn` | Glenn SQLite database format |
| `--claude` | Claude JSON export format |
| `--openclaw` | OpenClaw JSONL session format |
| `--source` | Path to source file/directory (optional for openclaw) |
| `--parallel` | Number of parallel workers (default: 10) |
| `--conversation` | Process specific conversation ID only (glenn only) |
| `--user` | Filter by username (glenn only) |
| `--limit` | Limit number of messages |

### `sync`
Main command for ongoing sync. Processes dirty items and writes files.

```bash
python cli.py sync -w ~/memory
python cli.py sync -w ~/memory --source ~/.openclaw/agents/main/sessions
python cli.py sync -w ~/memory --parallel 20
```

| Flag | Description |
|------|-------------|
| `-w`, `--workspace` | Workspace directory (required) |
| `--db` | Database filename (default: pyramid.db) |
| `--source` | Path to sessions directory for incremental import |
| `--parallel`, `-p` | Number of parallel workers (default: 10) |

**What sync does:**
1. If `--source` provided: incrementally import new messages from OpenClaw sessions
2. Assign unassigned observations to models
3. Create new tier-0 summaries (groups of 10 observations)
4. Create higher-tier summaries (groups of 10 lower-tier summaries)
5. Process dirty summaries (regenerate if inputs changed)
6. Embed new observations and summaries
7. Synthesize dirty models
8. Write markdown files to workspace

### `search`
Semantic search across memory with optional temporal weighting.

```bash
python cli.py search -w ~/memory "What programming languages does the user prefer?"
python cli.py search -w ~/memory "user's family" --limit 10 --raw
python cli.py search -w ~/memory "recent projects" --time-weight 0.5  # favor recent results
python cli.py search -w ~/memory "historical facts" --time-weight 0   # pure semantic
```

| Flag | Description |
|------|-------------|
| `-w`, `--workspace` | Workspace directory (required) |
| `--db` | Database filename (default: pyramid.db) |
| `--limit` | Number of results (default: 20) |
| `--raw` | Show raw results without LLM synthesis |
| `--time-weight` | Time decay weight from 0-1 (default: 0.3). 0 = pure semantic similarity, 1 = heavy recency bias. |

### Internal Commands

For debugging and manual control. Accessed via `cli.py internal COMMAND`.

#### `internal observe`
Add a single observation manually. Note: single observations are assigned in small batches, limiting the system's ability to discover new models. Prefer bulk imports.

```bash
python cli.py internal observe -w ~/memory "User prefers vim keybindings"
```

#### `internal summarize`
Run summarization only.

```bash
python cli.py internal summarize -w ~/memory
python cli.py internal summarize -w ~/memory --max-obs 100  # Limit observations
python cli.py internal summarize -w ~/memory --max-tier 1   # Only build up to tier 1
```

#### `internal embed`
Generate embeddings only.

```bash
python cli.py internal embed -w ~/memory
python cli.py internal embed -w ~/memory --force  # Re-embed everything
```

#### `internal generate`
Write markdown files from cached synthesis (no LLM calls).

```bash
python cli.py internal generate -w ~/memory
```

#### `internal synthesize`
Synthesize dirty models without writing files.

```bash
python cli.py internal synthesize -w ~/memory
```

## Output Files

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

## Module Reference

### `db.py`
SQLAlchemy models and database initialization.

- `Model`, `Observation`, `Summary`, `SummarySource`, `ImportedSession` - ORM classes
- `get_engine(db_path)` - Create SQLAlchemy engine
- `get_session(db_path)` - Create session
- `init_db(db_path)` - Initialize tables, run migrations, create base models
- `migrate_db(db_path)` - Add new columns to existing databases

### `llm.py`
LLM integration for observation extraction.

- `client` - OpenAI client instance
- `MODEL` - Model name constant
- `estimate_tokens(text)` - Token count estimation
- `chunk_messages(messages)` - Split messages into processable chunks
- `process_chunk(chunk)` - Extract observations from a chunk
- `extract_observations(messages, on_progress, max_workers)` - Main extraction entry point

### `summarize.py`
Summarization pipeline with dirty tracking.

- `STEP` - Items per summary (10)
- `assign_models_to_observations(session, observations)` - Model assignment
- `mark_model_dirty(session, model_id)` - Mark model for re-synthesis
- `mark_overlapping_summaries_dirty(session, model_id, timestamp)` - Mark affected summaries
- `record_summary_sources(session, summary, source_type, source_ids)` - Track what went into a summary
- `propagate_dirty_upward(session, summary)` - Mark parent summaries dirty
- `run_tier0_summarization(db_path, on_progress, max_workers)` - Run tier 0
- `run_higher_tier_summarization(db_path, on_progress, max_workers)` - Run tiers 1+
- `process_dirty_tier0(db_path, on_progress, max_workers)` - Regenerate dirty tier-0 summaries
- `process_dirty_higher_tiers(db_path, on_progress, max_workers)` - Regenerate dirty higher-tier summaries
- `process_all_dirty(db_path, on_progress, max_workers)` - Process all dirty summaries

### `pyramid.py`
Pyramid retrieval and synthesis.

- `get_pyramid(session, model_id)` - Returns dict of tier → summaries
- `get_unsummarized_observations(session, model_id, by_tier)` - Get observations not yet in tier 0
- `get_non_overlapping_summaries(by_tier)` - Filter summaries to avoid tier overlap
- `synthesize_model(name, description, by_tier, unsummarized_obs)` - Generate coherent mental model narrative
- `prepare_model_data(session, model, ref_date)` - Prepare data for synthesis
- `synthesize_dirty_models(db_path, on_progress, max_workers)` - Synthesize all dirty models, cache results

### `embeddings.py`
Vector embedding utilities.

- `EMBEDDING_MODEL` - Model name
- `EMBEDDING_DIM` - Dimension count (1536)
- `enrich_for_embedding(text, timestamp, end_timestamp)` - Prepend temporal context
- `embed_many(texts, max_workers, on_progress)` - Batch embed with parallel processing
- `search_memory(conn, query_text, limit, time_weight)` - Search memory with temporal reranking

### `loaders.py`
Message loading from various formats.

- `load_glenn_messages(source, conversation, user, limit)` - Load from Glenn SQLite format
- `load_claude_messages(source, limit)` - Load from Claude JSON export
- `load_openclaw_messages(source, limit)` - Load from OpenClaw JSONL sessions
- `load_openclaw_incremental(source, session_tracking)` - Load only new messages since last sync

### `generate.py`
Markdown generation from cached synthesis.

- `CORE_MODELS` - List of core model names (assistant, user)
- `update_model_descriptions(session, on_progress)` - Fill in missing descriptions
- `render_memory(assistant_content, user_content, other_models)` - Generate MEMORY.md content
- `render_model_file(model)` - Generate model file from cached synthesis
- `export_models(workspace, db_path, on_progress, model_ids)` - Main export function

### `sync.py`
Orchestration for the sync command.

- `embed_new_items(db_path, on_progress, max_workers)` - Embed items without embeddings
- `write_model_files(db_path, workspace, on_progress)` - Write markdown files
- `sync(workspace, db, source, on_progress, max_workers)` - Main sync function

### `cli.py`
Command-line interface with 4 main commands and internal subgroup.

## Processing Flows

### Sync Flow
```
sync command
├── If --source: load_openclaw_incremental → extract_observations → save to DB
├── run_tier0_summarization (assign + create new tier-0 summaries)
├── run_higher_tier_summarization (create new higher-tier summaries)
├── process_all_dirty (regenerate dirty summaries)
├── embed_new_items (embed new observations/summaries)
├── synthesize_dirty_models (synthesize dirty models, cache results)
└── export_models (write markdown files from cached synthesis)
```

### Dirty Propagation Flow
```
New observation added
└── Model marked content_dirty = True

Summary regenerated
├── Record sources in summary_sources
├── Parent summaries marked is_dirty = True
└── Model marked content_dirty = True

Model synthesized
├── Result cached in synthesized_content
└── content_dirty = False
```

### Search Flow
```
Query → get_embedding → memory_vec MATCH → candidates (3x limit)
     → fetch timestamps → compute_time_penalty → rerank by combined score
     → fetch Observation/Summary objects → LLM synthesis → answer
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

**Note for agents**: `pip install -r requirements.txt` handles everything including sqlite-vec. The extension is automatically loaded when needed. No manual SQLite compilation required.
