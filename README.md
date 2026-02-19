# Code Awareness

Build a developer's mental model of any codebase from its git history.

Instead of reading every file, this tool processes git commits (messages + diffs) through an LLM to extract architectural observations, organizes them into naturally-emerging subsystem models, and synthesizes them into compressed, temporally-organized narratives. The result reads like notes from a developer who's been on the project for months — not documentation, but *understanding*.

## What it produces

From a repo's git history, you get:

```
workspace/
  MEMORY.md              # Index of all models
  models/
    server.md            # Backend: routes, middleware, streaming, state
    client.md            # Frontend: components, state management, rendering
    database.md          # Schema, migrations, ORM patterns
    deployment.md        # Build, hosting, service management
    architecture.md      # Cross-cutting design decisions
    memory-system.md     # (emergent) — discovered from commit patterns
    auth.md              # (emergent) — discovered from commit patterns
    ...
```

**Base models** (architecture, server, client, database, deployment) are seeded to give the LLM reasonable starting categories. **Additional models emerge naturally** as the LLM discovers distinct subsystems in the commit history — things like `memory-system`, `streaming`, `auth`, or `twitter-api` get their own files automatically.

Each model is organized by time:
- **Last 3 Days** — full detail on recent changes
- **This Week / Month / Quarter / Year** — progressively compressed
- **Earlier** — broad strokes

This means recent work is vivid and older decisions fade to their essentials — like how a developer actually remembers a project.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenAI API key
echo "OPENAI_API_KEY=sk-..." > .env
# or: export OPENAI_API_KEY=sk-...

# 3. Run the full pipeline on any git repo
python cli.py import -w ./my-project-awareness --git --source /path/to/repo

# 4. Summarize, synthesize, and generate markdown
python cli.py internal summarize -w ./my-project-awareness
python cli.py internal synthesize -w ./my-project-awareness
python cli.py internal generate -w ./my-project-awareness

# 5. Read the output
cat ./my-project-awareness/MEMORY.md
ls ./my-project-awareness/models/
```

Or use `git-sync` to do it all in one command:

```bash
python cli.py git-sync -w ./my-project-awareness --repo /path/to/repo
```

### Ongoing updates

After new commits are pushed:

```bash
python cli.py git-sync -w ./my-project-awareness --repo /path/to/repo
```

This incrementally processes only new commits since the last sync, updates affected summaries, and re-synthesizes changed models. The system tracks the last processed commit automatically.

## How it works

```
Git log (commits + diffs)
    ↓
LLM extracts architectural observations
    ("Messages are sent via POST to stage the prompt, then GET opens
     the SSE stream — a two-step pattern to avoid URL length limits")
    ↓
Observations assigned to models (server, client, database, ...)
    ↓
Pyramidal compression: 10 observations → tier-0 summary
                       10 tier-0s → tier-1 summary
                       10 tier-1s → tier-2 summary ...
    ↓
Synthesis: pyramid + recent observations → temporal narrative per model
    ↓
Markdown files an agent can load as context
```

**Key design choice**: Git commits aren't conversations, but they carry something conversations don't — the *reasoning behind changes*. A commit message says what was done and why. The diff shows how. Together, they carry the kind of understanding that only a developer who's been on the project has.

## What this is for

Feed the generated markdown into an AI coding agent's context to give it genuine codebase awareness. Instead of the agent reading files from scratch every session, it starts with a synthesized understanding of:

- How the system is architected and why
- What patterns are used and where
- What changed recently vs. what's been stable
- Where the fragile areas are
- How components connect to each other

This is different from documentation. Documentation describes the intended state. Code awareness describes the *actual* state — including the dead ends, the refactors, and the design decisions that only show up in the commit history.

## CLI Reference

### Main Commands

| Command | Description |
|---------|-------------|
| `import --git --source /path/to/repo` | Import git history and extract observations |
| `git-sync --repo /path/to/repo` | Full pipeline: import → summarize → synthesize → generate |

### Import options

```bash
python cli.py import -w WORKSPACE --git --source /path/to/repo \
    --limit 100       # Process only N most recent commits
    --since 2025-01-01  # Only commits after this date
    --parallel 10     # LLM workers (default: 10)
```

### Internal commands (for debugging/manual control)

```bash
python cli.py internal summarize -w WORKSPACE    # Assign observations + create summaries
python cli.py internal synthesize -w WORKSPACE   # Synthesize models from pyramid
python cli.py internal generate -w WORKSPACE     # Write markdown files
python cli.py internal embed -w WORKSPACE        # Generate embeddings for search
python cli.py internal observe -w WORKSPACE "observation text"  # Add manually
```

### Search

```bash
python cli.py search -w WORKSPACE "how does authentication work?"
python cli.py search -w WORKSPACE "streaming" --raw  # Show raw results
```

## Configuration

Environment variables via `.env`:
```
OPENAI_API_KEY=sk-...
```

The pipeline uses `gpt-4.1-mini` for observation extraction, model assignment, summarization, and synthesis. Embeddings use `text-embedding-3-small`.

## Testing

```bash
OPENAI_API_KEY=fake pytest test/ -v  # 76 tests, no API calls needed
```

## How it compares to existing approaches

| Approach | What it gives you |
|----------|------------------|
| **Reading files** | Current state, no history or reasoning |
| **RAG over code** | Fragments matched by similarity, no coherent picture |
| **Documentation** | Intended state, often outdated |
| **Code awareness** | Synthesized understanding from actual evolution, organized by subsystem and time |

## Origin

This is a branch of [Pyramid Memory](https://github.com/finereli/pyramid), a hierarchical memory system for AI agents. Pyramid Memory builds mental models of *people* from conversations. Code Awareness applies the same philosophy — pyramidal compression, temporal weighting, emergent topic models — to build mental models of *codebases* from git history.

## Dependencies

```
openai          # LLM and embeddings
click           # CLI
python-dotenv   # .env loading
sqlalchemy      # ORM
sqlite-vec      # Vector search (installed via pip)
pytest          # Testing
```
