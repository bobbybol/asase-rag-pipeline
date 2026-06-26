"""
RAG Service
Hybrid retrieval (BM25 + pgvector) with Reciprocal Rank Fusion.
Backed by Supabase pgvector for persistence.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_community.retrievers import BM25Retriever

logger = logging.getLogger(__name__)


# === Reciprocal Rank Fusion ===

def _rrf_fuse(
    results_a: list[Document],
    results_b: list[Document],
    weight_a: float = 0.5,
    weight_b: float = 0.5,
    k: int = 60,
    top_n: int = 5,
) -> list[Document]:
    """
    Combine two ranked result lists using Reciprocal Rank Fusion.

    RRF score for a document = sum of weight / (rank + k) across all retrievers.
    A higher combined score means the document ranked well in both lists.
    k=60 is the standard constant that dampens the impact of very high ranks.
    """
    scores: dict[str, tuple[float, Document]] = {}

    for rank, doc in enumerate(results_a):
        key = doc.page_content
        rrf_score = weight_a * (1.0 / (rank + k))
        if key in scores:
            scores[key] = (scores[key][0] + rrf_score, doc)
        else:
            scores[key] = (rrf_score, doc)

    for rank, doc in enumerate(results_b):
        key = doc.page_content
        rrf_score = weight_b * (1.0 / (rank + k))
        if key in scores:
            scores[key] = (scores[key][0] + rrf_score, doc)
        else:
            scores[key] = (rrf_score, doc)

    ranked = sorted(scores.values(), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]


# === RAG Service ===

class RAGService:
    """
    Production RAG retrieval layer.

    Combines:
    - Supabase pgvector for persistent semantic (vector) search
    - In-memory BM25 for keyword search
    - RRF fusion for hybrid retrieval

    The BM25 index is rebuilt from pgvector on startup and updated on every
    ingestion so keyword and vector search stay in sync.
    """

    EMBEDDING_MODEL = "gemini-embedding-2-preview"

    def __init__(self, database_url: str, collection_name: str, k: int = 5):
        self._database_url = database_url
        self._collection_name = collection_name
        self._k = k

        self._embeddings = GoogleGenerativeAIEmbeddings(model=self.EMBEDDING_MODEL)
        self._vectorstore: Optional[PGVector] = None
        self._bm25: Optional[BM25Retriever] = None
        self._bm25_docs: list[Document] = []

    # --- Lazy vectorstore init ---

    @property
    def vectorstore(self) -> PGVector:
        if self._vectorstore is None:
            self._vectorstore = PGVector(
                embeddings=self._embeddings,
                collection_name=self._collection_name,
                connection=self._database_url,
                use_jsonb=True,
            )
        return self._vectorstore

    # --- Startup: load existing docs to warm up BM25 ---

    def load_existing_docs(self) -> int:
        """
        Fetch all documents from pgvector and build the BM25 index.
        Called once at startup so keyword search works without a cold start.
        Returns the number of documents loaded.
        """
        try:
            # PGVector exposes a similarity search; we use a broad query
            # to pull representative documents. For a true full scan, use
            # the underlying store's get() method if available.
            results = self.vectorstore.similarity_search(
                query="electricity metering solar mini-grid",
                k=200,  # generous upper bound for a course-scale collection
            )
            if results:
                self._bm25_docs = results
                self._bm25 = BM25Retriever.from_documents(results, k=self._k)
                logger.info(
                    "BM25 index warmed up from pgvector",
                    extra={"extra_data": {"doc_count": len(results)}},
                )
            else:
                logger.info("No existing documents found in pgvector — BM25 index empty")
            return len(results)
        except Exception as e:
            logger.warning(
                "Could not load existing docs for BM25 warm-up",
                extra={"extra_data": {"error": str(e)}},
            )
            return 0

    # --- Ingestion ---

    def add_documents(self, documents: list[Document]) -> list[str]:
        """
        Add documents to pgvector and rebuild the BM25 index.
        Returns the list of assigned document IDs.
        """
        ids = self.vectorstore.add_documents(documents)

        # Keep BM25 in sync: merge new docs into existing index
        self._bm25_docs = self._bm25_docs + documents
        self._bm25 = BM25Retriever.from_documents(self._bm25_docs, k=self._k)

        logger.info(
            "Documents ingested",
            extra={"extra_data": {"count": len(documents), "total_in_bm25": len(self._bm25_docs)}},
        )
        return ids

    # --- Retrieval ---

    def hybrid_search(self, query: str, k: Optional[int] = None) -> list[Document]:
        """
        Hybrid retrieval: BM25 (keyword) + pgvector (semantic) fused via RRF.

        If the BM25 index is empty (cold start before any ingestion), falls
        back gracefully to pure vector search.
        """
        top_n = k or self._k

        # Vector search (always available via Supabase)
        vector_results = self.vectorstore.similarity_search(query, k=top_n)

        # BM25 search (keyword) — skip if index is empty
        if self._bm25 is not None and self._bm25_docs:
            bm25_results = self._bm25.invoke(query)
            return _rrf_fuse(
                results_a=bm25_results,
                results_b=vector_results,
                weight_a=0.4,
                weight_b=0.6,
                top_n=top_n,
            )

        logger.debug("BM25 index empty — using vector-only retrieval")
        return vector_results[:top_n]

    @property
    def doc_count(self) -> int:
        """Number of documents currently in the BM25 index (proxy for total indexed)."""
        return len(self._bm25_docs)
