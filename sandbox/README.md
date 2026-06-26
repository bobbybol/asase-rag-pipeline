# Sandbox

Exploratory prototypes and investigation notes from building the RAG pipeline. These scripts exist to test isolated ideas before integrating them into the main application. Not production code — expect rough edges.

---

## Investigations

### Chunking strategy (`text_splitters.py`, `chunking/`)

Started with `RecursiveCharacterTextSplitter` (chunk size 500, overlap 50) which worked for most documents but produced context-poor fragments for dense technical standards text. The problem: a chunk reading *"The minimum crosstalk attenuation shall be 40 dB"* has no way of knowing it came from a section on power line communication unless the section header is included.

Tested semantic chunking (`semantic_chunking.py`) — embeds each sentence and splits when cosine similarity between adjacent sentences drops below a threshold. Better boundary detection for paragraph-level concepts, but 3–4× slower at indexing time and overkill for documents that are already well-structured.

Decision: stick with recursive chunking at 500 tokens / 10% overlap, but preserve document `title` and `section` in metadata so the agentic grader can weight source context.

Also prototyped `late_chunking.py` — embed the full document first, then chunk the embedding space. Interesting idea from the JinaAI paper but adds significant complexity and the gains weren't conclusive on technical prose. Parked.

---

### Embedding models (`embeddings_deep.py`)

Compared `text-embedding-004` and `gemini-embedding-2-preview` on a set of mini-grid retrieval queries. The preview model showed noticeably better separation for domain-specific terms (e.g. "SoC" correctly clustering with battery management content rather than social content). Went with `gemini-embedding-2-preview` throughout.

---

### Vector stores (`vector_stores.py`, `chroma_test.py`)

Prototyped with Chroma locally for speed, but local Chroma doesn't survive restarts cleanly when combined with a Docker-compose setup that remounts the volume. Switched to Supabase pgvector via `langchain-postgres`.

Key finding: `PGVector` with `use_jsonb=True` is significantly more flexible for metadata filtering than the default array storage — worth setting even if you don't use filters immediately.

Tested `pg_vector/supabase/supabase_connection.py` to verify connection string format and the HNSW index behaviour on Supabase's managed pgvector. The pooler connection string is needed when deploying outside Supabase's own network (i.e. from a local machine or a Render/Railway container that lacks IPv6).

---

### Hybrid search (`hybrid_search/`)

Pure semantic search struggled with exact-match queries — error codes like `E_CONN_REFUSED`, product SKUs, and standards references like `IEC 62055-41` have almost no semantic signature. BM25 handles these cleanly.

`prod_hybrid_search.py` tests manual Reciprocal Rank Fusion across a BM25 retriever and a vector retriever. RRF with `k=60` consistently outperformed either retriever alone on a mixed test set of semantic and keyword queries. The 40/60 BM25/semantic weight split was arrived at empirically — heavier semantic weighting helps for conceptual questions about mini-grid design while BM25 catches the specific references.

This is what ended up in `app/rag.py`.

---

### RAG chain basics (`rag_pipeline.py`)

Initial end-to-end RAG chain with a flat retriever and a simple prompt. Useful for establishing a baseline response quality before introducing the agentic loop. The retriever configuration `search_type="similarity", k=5` is what the production service uses, but wrapped inside the hybrid RRF layer.

---

### Advanced retrieval patterns (`advanced_rag.py`)

Tested three patterns that didn't make the final cut:

**Multi-Query Retriever** — generates several paraphrases of the original query and retrieves for each, then deduplicates. Adds 1–2 LLM calls per request; useful but the agentic rewrite loop achieves a similar outcome with better control over when the extra call is justified.

**Contextual Compression** — uses an LLM extractor to trim retrieved chunks down to only the sentences directly relevant to the query. Reduces context window usage but adds latency and cost per request. Overkill for this use case where chunks are already small.

**Parent Document Retriever** — indexes small child chunks for search but returns their larger parent chunk for generation. Good for long-form documents where a paragraph has the right keyword but the full section gives the real context. Worth revisiting if the knowledge base grows to include full technical manuals.

---

### Contextual retrieval (`contextual_retrieval.py`)

Investigated Anthropic's technique of prepending an LLM-generated context summary to each chunk before embedding. The retrieval quality improvement is real — especially for documents that use a lot of pronouns and implicit references. The trade-off is one LLM call per chunk at indexing time, which for a batch of 500 documents means meaningful cost and latency.

Decided against it for now. The mini-grid documents are already fairly self-contained at the chunk level. Can revisit if retrieval quality metrics suggest context loss is a problem.

---

### Agentic RAG (`agentic_rag.py`)

The core pattern that made it into production. A LangGraph `StateGraph` with a retrieve → grade → rewrite → retrieve loop. The grader outputs a relevance score per document; the router decides whether to accept, retry with a rewritten query, or fall back.

Key decision: use the same primary LLM for grading and query rewriting (Gemini Flash) rather than a smaller model. The cost delta is small and the grading accuracy is significantly better — using a weaker model for grading produced noisy scores that confused the router.

---

### Monitoring (`monitoring.py`)

Tested structured JSON logging with a custom `extra_data` field that LangSmith and log aggregators can index. The `RequestTimer` context manager is used throughout the main API to measure end-to-end latency per request. Metrics collector tracks token usage estimates, error rates, and cache hit ratios — surfaced via `GET /metrics`.

---

### Long context vs RAG (`long_contect_vs_rag.py`)

Quick test to understand when it is better to just stuff a long document into the context window versus retrieving relevant chunks. For the mini-grid domain with documents up to ~10 pages, chunked RAG wins on latency and cost. Long context becomes competitive only when the query genuinely requires reasoning across the entire document — not typical for the factual, lookup-style queries this system handles.

---

### Cost optimisation (`cost_optimization.py`)

Investigated caching at the embedding level (same query → skip re-embedding), model tiering (use Flash for grading/rewriting, reserve a larger model for generation), and response caching. The main app uses TTL-based response caching at the API layer which gives the biggest bang for the buck — cache hits add zero latency and zero cost.

---

### LangSmith tracing (`langsmith_setup.py`)

Verified the EU endpoint (`https://eu.api.smith.langchain.com`) and project configuration. Every `agent.invoke()` call is decorated with `@traceable`, and the FastAPI `/chat` endpoint is also traced so the full chain — security, cache check, retrieval, grading, generation — appears as a single trace in LangSmith.
