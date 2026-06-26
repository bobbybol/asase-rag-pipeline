"""
API Request and Response Models
Pydantic models for input validation and response structure.
"""

from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class ChatRequest(BaseModel):
    """Incoming chat request."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The user's message to the agent",
    )
    thread_id: str = Field(default="default", description="Conversation thread ID")


class RetrievedSource(BaseModel):
    """A single retrieved document returned alongside a chat response."""

    content: str
    source: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """Chat response returned to the client."""

    response: str
    thread_id: str
    model_used: str
    cached: bool = False
    processing_time_ms: float
    security_notes: list[str] = Field(default_factory=list)
    sources: list[RetrievedSource] = Field(
        default_factory=list,
        description="Documents retrieved to ground this response.",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    environment: str
    version: str = "1.1.0"
    checks: dict = {}


class MetricsResponse(BaseModel):
    """Metrics endpoint response."""

    total_requests: int
    total_errors: int
    error_rate: str
    avg_latency_ms: float
    cache_hit_rate: str
    total_input_tokens: int
    total_output_tokens: int


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str | None = None
    request_id: str | None = None


# === Ingestion ===

class DocumentInput(BaseModel):
    """A single document to be ingested into the vector store."""

    content: str = Field(
        ...,
        min_length=10,
        description="The text content of the document.",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata (source, topic, date, etc.).",
    )


class IngestRequest(BaseModel):
    """Request body for the /ingest endpoint."""

    documents: list[DocumentInput] = Field(
        ...,
        min_length=1,
        description="One or more documents to add to the knowledge base.",
    )


class IngestResponse(BaseModel):
    """Response returned after successful ingestion."""

    ingested_count: int
    collection: str
    total_indexed: int = Field(
        description="Total documents now in the BM25 index (approximate)."
    )