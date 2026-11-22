"""Strict Pydantic v2 contracts for RiverPay assistant outputs and KB metadata."""

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Reference to an official RiverPay knowledge base document and section."""

    doc: str = Field(..., description="Exact markdown filename, e.g., 03_fees_limits.md")
    section: str = Field(..., description="Exact markdown heading from the document")


class AssistantResponse(BaseModel):
    """Strict output contract for both batch evaluation and interactive chat."""

    question_id: str = Field(..., description="Question identifier, e.g., Q01 or CHAT-001")
    answer: str = Field(..., description="Plain-language answer for a customer")
    citations: list[Citation] = Field(
        default_factory=list,
        description="List of document/section citations supporting the answer",
    )
    refusal: bool = Field(
        ..., description="True if the assistant refused an out-of-scope, unanswerable, or disallowed request"
    )
    handoff_to_human: bool = Field(
        ..., description="True if the customer should be handed off to a human contact-center agent"
    )
    notes: str = Field(
        default="",
        description="Internal retrieval scores, active metadata filter, route taken, and latency_ms",
    )


class LLMGenerationPayload(BaseModel):
    """Structured JSON schema returned by the LLM before deterministic verification."""

    answer: str = Field(..., description="Plain-language customer response without internal filenames")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Citations from the provided context supporting every stated fact",
    )
    refusal: bool = Field(..., description="Whether the request is refused under assistant policy")
    handoff_to_human: bool = Field(..., description="Whether to hand off to a human agent")
    reasoning: str = Field(default="", description="Brief internal note on policy/conflict resolution")
