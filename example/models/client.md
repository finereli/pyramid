---
name: client
description: Frontend implementation — UI components, state management, rendering patterns, and user interactions
---

# Client

### This Month

The frontend client is tightly integrated with a multi-stage synchronization and data management process orchestrated through CLI commands. The core synchronization command, `sync`, consolidates previous disparate steps (e.g., `heartbeat`) into a single incremental operation that imports conversation session files, extracts observations, performs tiered summarizations, embeds new items, synthesizes models, and writes relevant markdown output files. This incremental process reads only changed data to minimize unnecessary work, allowing for efficient handling of large datasets and avoiding redundant file I/O by rewriting files only when their content has changed (via SHA-256 hashing).

The client manages its workspace around a directory containing a `pyramid.db` SQLite file alongside generated markdown models organized under `models/` and root-level identity files (`SOUL.md`, `USER.md`, `TOOLS.md`). The user and assistant memories are merged into `MEMORY.md` with distinct sections, while identity files remain hand-maintained and excluded from auto-generation.

Data ingestion supports multiple conversation export formats including Glenn SQLite, Claude JSON, and OpenClaw JSONL sessions. The client parses OpenClaw session chunks by reading new bytes only, efficiently syncing without full reloads. The frontend CLI allows both manual single-observation insertion (via the `observe` command) and bulk import/sync workflows, though batch processing is recommended for better model discovery and summarization.

Summary generation follows a tiered pyramid pattern, beginning with tier-0 daily summaries to curtail unassigned data growth and escalating to higher-tier summaries. This supports incremental, parallel execution of summarization tasks using Python’s `concurrent.futures.ThreadPoolExecutor` with configurable thread counts. Progress bars indicate completion status and distinguish between no-op and update cases to inform the user.

File output enforces consistent naming conventions for models, normalizing names to lowercase dash-separated strings to align with filesystem expectations and enable deterministic referencing. Exported markdown files embed YAML frontmatter metadata reflecting model identity and summarization tier.

Finally, recent frontend CLI refactoring consolidated commands related to extraction and generation into `import` and `generate` commands to reduce redundancy and simplify user interaction.

---

### This Week

(No new distinct information available for the last 3 days or this week beyond the This Month section.)
