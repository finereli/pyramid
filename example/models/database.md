---
name: database
description: Data layer — schema design, migrations, ORM usage, and storage patterns
---

# Database

### This Month  
The primary data store is a SQLite database named `pyramid.db`. It maintains tables for conversation data, observations, summaries, models, and imported sessions. The schema comprises ORM-mapped classes including `Model`, `Observation`, `Summary`, `ImportedSession`, and a newly introduced `summary_sources` table that tracks the provenance of synthesized summaries.  

`Observation` records are short, timestamped text facts linked optionally to a `Model`. They are created programmatically, commonly via LLM tool calls. The previous `importance` field on observations has been removed to simplify the schema, reflecting a shift away from ranking at the observation level. Summaries remain linked to models and include a `tier` field to classify their hierarchical level.  

The system uses embeddings generated via OpenAI’s embedding models for observations and summaries. These embeddings are stored in a SQLite virtual table implemented with `sqlite-vec`, enabling semantic nearest neighbor searches. This integration supports retrieval of relevant context for answer synthesis within LLM workflows.  

### This Quarter  
Prior to the schema refinements this month, the database design encompassed tables for observations, summaries, and models that contained text content, timestamps, an `importance` metric, and embedding vectors. Observations and summaries were associated with models to group them logically.  

Embeddings were embedded similarly using OpenAI API and stored in a SQLite virtual vector table (`sqlite-vec`), allowing similarity searches to retrieve related observations or summaries. This setup facilitated context-aware response generation by combining nearest neighbor retrieval with LLM prompt augmentation.  

### This Year  
The schema, as initially implemented, stored observations, summaries, and models in a SQLite database. Each table included text fields, timestamps, and an `importance` metric to express relative significance. Observations and summaries were assigned to models to organize the data.  

Embedding vectors were generated using OpenAI embeddings and persisted alongside the text data. A specialized virtual table leveraging `sqlite-vec` enabled nearest neighbor queries based on vector similarity. This design was intended to support semantic search capabilities for downstream LLM operations.  

---

The database model has evolved to remove the `importance` field from observations and introduce a provenance-tracking table (`summary_sources`). Embedding storage and retrieval via `sqlite-vec` remains consistent, supporting semantic search and context synthesis. The overall focus is on representing timestamped text observations and structured summaries linked to models, facilitating efficient retrieval and integration with LLM tools for automated data augmentation and response generation.
