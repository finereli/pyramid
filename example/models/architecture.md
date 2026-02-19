---
name: architecture
description: High-level system design — how major components connect, overall data flow, and key design decisions
---

# Architecture

### Last 3 Days  
The system uses a pyramid memory architecture that varies temporal resolution to emulate human memory: recent observations retain high detail while older data is progressively compressed into summarized tiers. Observations are extracted as discrete factual statements from conversations, organized hierarchically into tiers and temporally bucketed relative to the current day or reference date, enabling consistent time-based grouping.  

To support vague temporal queries, input text is enriched by prepending formatted date context before embedding, facilitating semantically aware temporal search across memories. Summaries across tiers are synthesized and deduplicated using specialized methods to prevent content overlap and maintain clarity.  

The architecture prioritizes preserving relationship-aware memory rather than simple retrieval, enabling the agent to maintain relational context across conversations and build coherent evolving knowledge structures.  

For database initialization, customizable base models seed mental model categories. In code-centric environments, five predefined git base models—architecture, server, client, database, and deployment—serve as foundational subsystems organizing observed knowledge. Mental models are generated via targeted voice prompts instructing the language model to produce dry, factual notes from a senior developer perspective, referencing specific files and functions, and avoiding marketing language.  

Mental models are stored as file-based categories named after subsystems (e.g., server.md). Emergent models are identified automatically from git commit patterns. The architectural strategy transforms git commit histories into temporally organized, synthesized developer mental models, enabling developers to grasp subsystem design and evolution without extensive codebase review.  

### This Month  
The pyramid memory architecture manages conversational observations through a multi-stage process documented in the README, including system design, data schemas, commands, and examples clarifying rationale. Core components align to functionality with specific modules: `cli.py` (command orchestration), `llm.py` (LLM interfacing and chunk processing), `summarize.py` (tiered summarization logic), `pyramid.py` (memory retrieval and synthesis), `export_models.py` (markdown export), and `db.py` (database models and ORM access).  

Multiple mental models capture derived descriptions and metadata (e.g., base flags), including assistant (self), user, and discovered entities. This replaced a prior restrictive system model to allow richer, more natural emergence of technical topics.  

Observations are temporally segmented into Last 3 Days, This Week, This Month, This Quarter, This Year, and Earlier to reflect human memory organization and improve narrative synthesis quality. Data compression occurs in tiers arranged in a pyramid, with each tier representing increasing temporal breadth and compression, facilitating scalable long-term memory.  

Importance is not evaluated at extraction time to avoid premature filtering; instead, importance judgements occur during summarization phases with full context, improving summary relevance and accuracy. Narrative output style varies by mental model: the assistant’s self-model uses first-person reflective summaries, while other models use coherent third-person prose for clarity.  

Semantic search incorporates temporal weighting via an adjustable exponential decay function (half-life default 30 days) controlled by a `--time-weight` parameter, balancing recency and semantic relevance during memory retrieval.  

### This Quarter  
The pyramid memory system structures Observations—short factual statements extracted from first-person conversations with timestamps—within mental models such as self, user, system, or dynamically discovered topics. These observations are assigned to mental models and grouped into a tiered pyramid structure where each higher level summarizes batches of ten lower items, representing exponentially increasing time spans.  

Mental models serve as categorical frameworks organizing observations and summaries. Permanent base models exist for self, user, and system, while topic-specific models emerge dynamically during summarization by GPT-4. Summaries use dense, telegram-style sentences preserving critical facts, augmented by importance flags (IMPORTANT, CRITICAL) for interpretability.  

The system pipeline consists of five core layers:  
- Extraction: Captures factual conversational data using GPT-4 with tool call annotations to ensure accuracy and categorization.  
- Organization: Assigns observations to appropriate mental models.  
- Compression: Summarizes observations into tiered pyramid levels, compressing older data progressively.  
- Synthesis: Produces narrative prose via GPT-4 summarization calls coordinated with ThreadPoolExecutor to maximize throughput.  
- Retrieval: Retrieves compressed, synthesized mental model narratives for ongoing context.  

Prompt engineering was standardized to enforce narrative style and consistent importance notation, improving summary readability and coherence. Export stages perform deduplication and merge sparse observations into unified mental model narratives suitable for human consumption.  

### This Year  
The pyramid memory architecture focuses on long-term conversational context retention by hierarchically organizing and compressing factual observations into coherent mental models and summaries. This design supports a nuanced, relationally aware memory rather than simple fact retrieval.  

Observed data is structured into mental models representing entities such as assistant self, user, system components, and emergent technical topics. These mental models form the backbone for summarization and retrieval, enabling meaningful comprehension of system subsystems and conversational context over time.  

The use of tiered, exponentially compressed summary levels avoids memory bloat and supports efficient, scalable long-term memory management. The architecture’s integration with git commit histories allows creation and evolution tracking of mental models aligned with codebase development, providing developers with synthesized, time-ordered design insights without exhaustive code reading.  

Prompt-based narrative generation ensures output mimics senior developer technical notes, referencing concrete files, functions, and architectural decisions free from promotional language. This supports a developer-centric understanding of complex systems driven by observable data rather than abstract summaries.
