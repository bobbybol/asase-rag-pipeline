"""
Agentic RAG with LangGraph

Replaces the naive LLM pass-through with a self-correcting retrieval loop:

  START → retrieve → grade → generate   (good relevance)
                           → rewrite → retrieve (loop, low relevance)
                           → fallback   (out of retries / no docs)

The primary/fallback LLM structure is preserved for the generation step.
"""

from typing import Optional, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from app.config import get_settings
from app.rag import RAGService


# === State ===

class RAGState(TypedDict):
    query: str
    rewritten_query: str
    documents: list[Document]
    generation: str
    relevance_score: float
    retry_count: int


# === Agent ===

class ProductionAgent:
    """
    Agentic RAG agent built on LangGraph.

    Retrieves from the hybrid RAG pipeline, grades relevance, rewrites the
    query if needed, and generates a grounded answer. Falls back gracefully
    when retrieval fails after all retries.

    Public interface is unchanged from the original agent:
        agent.invoke(message) -> {"response": str, "model_used": str, "sources": list}
    """

    def __init__(self, rag_service: RAGService):
        settings = get_settings()
        self._rag = rag_service

        self._primary_llm = ChatGoogleGenerativeAI(
            model=settings.primary_model,
            temperature=0,
            timeout=30,
            max_retries=0,
            api_key=settings.google_api_key,
        )
        self._fallback_llm = ChatGoogleGenerativeAI(
            model=settings.fallback_model,
            temperature=0,
            timeout=30,
            max_retries=0,
            api_key=settings.google_api_key,
        )
        self._relevance_threshold = settings.rag_relevance_threshold
        self._max_retries = settings.max_retries
        self._graph = self._build_graph()

    # ------------------------------------------------------------------ #
    # Node functions
    # ------------------------------------------------------------------ #

    def _retrieve(self, state: RAGState) -> dict:
        """Hybrid search: BM25 + pgvector via RRF."""
        query = state.get("rewritten_query") or state["query"]
        documents = self._rag.hybrid_search(query)
        return {"documents": documents}

    def _grade(self, state: RAGState) -> dict:
        """
        Ask the LLM to score each retrieved document's relevance (0–1).
        Irrelevant documents are filtered out; average score drives routing.
        """
        query = state["query"]
        documents = state["documents"]

        if not documents:
            return {"documents": [], "relevance_score": 0.0}

        grading_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a relevance grader for a RAG system about smart electricity "
                "metering, mini-grids, and rural solar electrification.\n\n"
                "Given a user query and a document excerpt, output ONLY a single number "
                "between 0 and 1:\n"
                "  1.0 = directly answers the query\n"
                "  0.7 = contains related information\n"
                "  0.3 = tangentially related\n"
                "  0.0 = not relevant\n\n"
                "Output ONLY the number, nothing else.",
            ),
            (
                "human",
                "Query: {query}\n\nDocument: {document}\n\nRelevance score (0–1):",
            ),
        ])

        scores: list[float] = []
        relevant_docs: list[Document] = []

        for doc in documents:
            chain = grading_prompt | self._primary_llm
            try:
                result = chain.invoke({"query": query, "document": doc.page_content})
                score = float(result.content.strip())
            except (ValueError, Exception):
                score = 0.5

            scores.append(score)
            if score >= self._relevance_threshold:
                relevant_docs.append(doc)

        avg_score = sum(scores) / len(scores)
        return {"documents": relevant_docs, "relevance_score": avg_score}

    def _rewrite(self, state: RAGState) -> dict:
        """Reformulate the query to improve retrieval on the next attempt."""
        rewrite_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a query rewriter for a RAG system about mini-grids and rural "
                "electrification. The original query did not retrieve relevant documents.\n\n"
                "Rewrite the query to be more specific and likely to match technical "
                "documentation. Add synonyms, expand abbreviations, or rephrase to match "
                "how engineering documents are written.\n\n"
                "Output ONLY the rewritten query, nothing else.",
            ),
            ("human", "Original query: {query}\n\nRewritten query:"),
        ])

        chain = rewrite_prompt | self._primary_llm
        result = chain.invoke({"query": state["query"]})
        return {
            "rewritten_query": result.content.strip(),
            "retry_count": state["retry_count"] + 1,
        }

    def _generate(self, state: RAGState) -> dict:
        """Generate a grounded answer from retrieved documents using the primary LLM."""
        context = "\n\n".join(
            f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in state["documents"]
        )

        generate_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a knowledgeable assistant specialising in smart electricity "
                "metering, mini-grids, and rural solar electrification in Africa.\n\n"
                "Answer the question using ONLY the provided context. Be concise and "
                "precise. Cite sources by mentioning the document name in brackets where "
                "relevant. If the context is insufficient, say so clearly.",
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:",
            ),
        ])

        for llm, label in ((self._primary_llm, "primary"), (self._fallback_llm, "fallback")):
            try:
                chain = generate_prompt | llm
                result = chain.invoke({"context": context, "query": state["query"]})
                return {"generation": result.content, "model_used": label}
            except Exception:
                continue

        return {
            "generation": "I'm sorry, I encountered an error generating a response. Please try again.",
            "model_used": "error_handler",
        }

    def _fallback(self, state: RAGState) -> dict:
        """Graceful response when retrieval fails after all retries."""
        return {
            "generation": (
                "I couldn't find relevant information to answer your question. "
                "This topic may not yet be covered in the knowledge base. "
                "You can add documents via the /ingest endpoint and try again."
            ),
            "model_used": "fallback_handler",
        }

    # ------------------------------------------------------------------ #
    # Routing
    # ------------------------------------------------------------------ #

    def _route(self, state: RAGState) -> Literal["generate", "rewrite", "fallback"]:
        score = state.get("relevance_score", 0.0)
        retries = state.get("retry_count", 0)
        docs = state.get("documents", [])

        if score >= self._relevance_threshold and docs:
            return "generate"
        if retries < self._max_retries:
            return "rewrite"
        return "fallback" if not docs else "generate"

    # ------------------------------------------------------------------ #
    # Graph construction
    # ------------------------------------------------------------------ #

    def _build_graph(self):
        graph = StateGraph(RAGState)

        graph.add_node("retrieve", self._retrieve)
        graph.add_node("grade", self._grade)
        graph.add_node("rewrite", self._rewrite)
        graph.add_node("generate", self._generate)
        graph.add_node("fallback", self._fallback)

        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "grade")
        graph.add_conditional_edges(
            "grade",
            self._route,
            {"generate": "generate", "rewrite": "rewrite", "fallback": "fallback"},
        )
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("generate", END)
        graph.add_edge("fallback", END)

        return graph.compile()

    # ------------------------------------------------------------------ #
    # Public interface (unchanged from original ProductionAgent)
    # ------------------------------------------------------------------ #

    @traceable(name="production_agent_invoke")
    def invoke(self, message: str) -> dict:
        """
        Invoke the agentic RAG pipeline.
        Returns: {"response": str, "model_used": str, "sources": list[dict]}
        """
        initial_state: RAGState = {
            "query": message,
            "rewritten_query": "",
            "documents": [],
            "generation": "",
            "relevance_score": 0.0,
            "retry_count": 0,
        }

        result = self._graph.invoke(initial_state)

        sources = [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source"),
                "metadata": doc.metadata,
            }
            for doc in result.get("documents", [])
        ]

        return {
            "response": result.get("generation", ""),
            "model_used": result.get("model_used", "unknown"),
            "sources": sources,
        }
