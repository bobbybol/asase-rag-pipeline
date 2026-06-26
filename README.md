# asase-rag-pipeline

A production-grade **Agentic RAG** pipeline built with LangGraph, backed by Supabase pgvector and hybrid BM25 + semantic retrieval. The domain is smart electricity metering and mini-grid deployment in rural Africa.

---

## What this is

This repository implements a complete retrieval-augmented generation system as a FastAPI service. The core idea: instead of a naive retrieve-then-generate loop, the agent _evaluates_ what it retrieves, rewrites the query if relevance is low, and retries — only generating an answer when it has grounding it can trust.

The knowledge base covers technical material on mini-grids, prepaid metering (STS/PAYG), solar sizing, battery storage, and rural electrification in Sub-Saharan Africa.

---

## Architecture

```
POST /chat
    │
    ├─ Rate limiter          (slowapi — 20 req/min per IP, returns 429 on breach)
    │
    ├─ LangSmith trace       (wraps the full request for end-to-end observability)
    │
    ├─ Input security        (prompt injection detection, PII masking)
    │   └─ 400 if blocked
    │
    ├─ Response cache        (TTL-based, keyed on cleaned query — returns early on hit)
    │
    ├─ AgenticRAG (LangGraph)
    │       │
    │       ├─ retrieve      Hybrid search: BM25 + pgvector, fused via RRF
    │       ├─ grade         LLM scores each retrieved doc 0–1 for relevance
    │       ├─ rewrite ──►   If score < threshold and retries remain,
    │       │   └─ retrieve  reformulate query and search again
    │       ├─ generate      Grounded answer from primary LLM (Gemini Flash)
    │       │                Falls back to secondary model on failure
    │       └─ fallback      Graceful message when retrieval fails completely
    │
    ├─ Output security       (validates and sanitises the generated response)
    │
    ├─ Cache store           (write validated response for future cache hits)
    │
    ├─ Metrics               (latency, token estimates, error rate, cache hit rate)
    │
    └─ Structured log        (thread ID, model used, latency, sources retrieved)
            │
            ▼
        JSON response        (response, sources, model_used, processing_time_ms, …)
```

Retrieved sources are returned alongside every response so callers can see exactly what grounded the answer.

---

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI, slowapi (rate limiting) |
| Orchestration | LangGraph (StateGraph) |
| LLM | Google Gemini 2.5 Flash (primary + fallback) |
| Embeddings | Google `gemini-embedding-2-preview` |
| Vector store | Supabase pgvector (`langchain-postgres`) |
| Keyword search | BM25 (`rank-bm25`) |
| Retrieval fusion | Reciprocal Rank Fusion (RRF) |
| Tracing | LangSmith |
| Config | pydantic-settings |
| Containerisation | Docker + docker-compose |

---

## Project structure

```
app/
├── main.py        FastAPI app — endpoints, lifespan, wiring
├── rag.py         RAGService — hybrid retrieval, ingestion, BM25 management
├── agent.py       ProductionAgent — agentic RAG LangGraph graph
├── config.py      Pydantic settings (env-validated)
├── models.py      Request / response schemas
├── security.py    Input sanitisation, PII masking, output validation
├── cache.py       TTL response cache
└── monitoring.py  Structured logging, metrics, request timer

scripts/
└── seed.py        Populate the knowledge base with mini-grid domain documents

sandbox/
└── ...            Exploratory prototypes used during development
```

---

## Setup

### 1. Prerequisites

- Python 3.13+
- A [Supabase](https://supabase.com) project with pgvector enabled
- A Google AI API key (for Gemini models and embeddings)
- A LangSmith account (optional, for tracing)

### 2. Install dependencies

```bash
pip install uv
uv sync
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
GOOGLE_API_KEY=your_google_api_key
SUPABASE_DATABASE_URL=postgresql://postgres:[password]@db.[ref].supabase.co:5432/postgres

# Optional
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=asase-rag-pipeline
```

### 4. Seed the knowledge base

```bash
python scripts/seed.py
```

This embeds 16 technical documents into the Supabase pgvector collection `minigrid_docs`. Takes ~30 seconds.

### 5. Run the API

```bash
uvicorn app.main:app --reload
```

Or with Docker:

```bash
docker-compose up
```

---

## API endpoints

### `POST /chat`
Query the RAG pipeline.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What battery technology works best for remote mini-grids?"}'
```

Response includes `response`, `sources` (retrieved documents), `model_used`, and `processing_time_ms`.

### `POST /ingest`
Add documents to the knowledge base at runtime.

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "content": "Your technical document text here.",
        "metadata": {"source": "my_report.pdf", "topic": "grid_design"}
      }
    ]
  }'
```

### `GET /health`
Returns status of all components including RAG service.

### `GET /metrics`
Request counts, error rate, latency, token usage, cache hit rate.

### `GET /cache/stats`
Cache performance breakdown.

---

## How the hybrid search works

Every query runs through two retrieval paths in parallel:

1. **Semantic search** — the query is embedded and compared against document vectors in Supabase pgvector. Finds conceptually related content even when exact terms differ.
2. **BM25 keyword search** — an in-memory BM25 index (rebuilt from pgvector on startup and after each ingest) finds documents containing the literal query terms. Effective for product codes, standards references (e.g. IEC 62055-41, DLMS/COSEM), and proper nouns.

Results from both paths are fused using **Reciprocal Rank Fusion (RRF)** with a 40/60 weight split (BM25/semantic), producing a single ranked list that outperforms either approach alone.

---

## How the agentic loop works

After retrieval, an LLM grades each document's relevance on a 0–1 scale. If the average score falls below the configured threshold (`RAG_RELEVANCE_THRESHOLD`, default `0.5`):

- If retries remain (`MAX_RETRIES`, default `3`): the query is rewritten to be more specific, and retrieval runs again.
- If retries are exhausted with no relevant documents: a fallback response is returned.
- If relevant documents are found at any point: a grounded answer is generated.

This self-correcting loop handles ambiguous queries and jargon variations without manual query engineering.
