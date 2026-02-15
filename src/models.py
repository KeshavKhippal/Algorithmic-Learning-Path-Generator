"""
Pydantic models for the Algorithmic Learning Path Generator.

Phase 1: resource records, ingestion results, ingestion summaries.
Phase 2: extracted concepts, canonical concepts, phase2 summary.
"""

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# =========================================================================
# Phase 1 Literals & Models
# =========================================================================

ContentType = Literal["article", "youtube", "unknown"]
Status = Literal[
    "ok",
    "no_content",
    "no_transcript",
    "transcript_disabled",
    "rate_limited",
    "video_unavailable",
    "network_error",
    "failed",
    "skipped",
]

DifficultyBucket = Literal["beginner", "intermediate", "advanced"]


class ResourceRecord(BaseModel):
    """Mirrors a single row of the ``RawResources`` table."""

    id: Optional[int] = None
    url: str
    content_type: ContentType
    title: Optional[str] = None
    raw_text: Optional[str] = None
    status: Status
    extracted_at: Optional[datetime] = None
    notes: Optional[str] = None


class IngestResult(BaseModel):
    """Result of ingesting a single URL."""

    url: str
    status: Status
    title: Optional[str] = None
    error: Optional[str] = None


class IngestSummary(BaseModel):
    """Aggregated ingestion summary that gets serialised to JSON."""

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    no_content: int = 0
    no_transcript: int = 0
    transcript_disabled: int = 0
    rate_limited: int = 0
    video_unavailable: int = 0
    network_error: int = 0
    skipped: int = 0
    errors: List[str] = Field(default_factory=list)


# =========================================================================
# Phase 2 Models
# =========================================================================


class CandidateConcept(BaseModel):
    """A single candidate concept extracted from a resource (pre-clustering)."""

    text: str
    sentence: str
    resource_id: int


class ExtractedConceptRecord(BaseModel):
    """Mirrors a single row of the ``ExtractedConcepts`` table."""

    id: Optional[int] = None
    resource_id: int
    concept: str
    canonical_id: Optional[int] = None
    sentence: Optional[str] = None
    embedding: Optional[bytes] = None
    created_at: Optional[datetime] = None


class CanonicalConceptRecord(BaseModel):
    """Mirrors a single row of the ``CanonicalConcepts`` table."""

    id: Optional[int] = None
    canonical_concept: str
    difficulty_score: Optional[float] = None
    difficulty_bucket: Optional[DifficultyBucket] = None
    example_sentence: Optional[str] = None
    resource_count: Optional[int] = None
    created_at: Optional[datetime] = None


class Phase2Summary(BaseModel):
    """Summary written to ``phase2_summary.json``."""

    resources_processed: int = 0
    candidate_concepts: int = 0
    canonical_concepts: int = 0
    avg_concepts_per_resource: float = 0.0
    difficulty_distribution: Dict[str, int] = Field(
        default_factory=lambda: {"beginner": 0, "intermediate": 0, "advanced": 0}
    )
