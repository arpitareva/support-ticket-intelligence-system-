"""Response models for the REST API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.query_plan import QueryPlan


class HealthResponse(BaseModel):
    status: str = Field(examples=["healthy"])
    database: str
    tickets_loaded: int
    llm_provider: str
    llm_configured: bool
    dataset_range: dict[str, datetime | None] = Field(default_factory=dict)
    anomaly_reference_time: datetime | None = None
    version: str


class ExecutionMeta(BaseModel):
    llm_provider: str
    llm_model: str | None = None
    llm_latency_ms: int | None = None
    query_latency_ms: int | None = None
    rows_returned: int = 0
    row_limit_applied: int | None = None
    sql: str | None = None
    llm_repair_attempts: int = 0


class QueryResponse(BaseModel):
    question: str
    answer: str
    query_plan: QueryPlan
    plan_explanation: str
    results: list[dict[str, Any]] = Field(default_factory=list)
    # Populated for two-stage (group_detail) answers: the ranked group summary
    # that selected the entity whose rows appear in ``results``.
    groups: list[dict[str, Any]] = Field(default_factory=list)
    result_type: str
    execution: ExecutionMeta


class Anomaly(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_id: str | None = None
    entity_type: str = "ticket"  # "ticket" | "agent"
    entity_id: str
    anomaly_type: str
    severity: str  # low | medium | high | critical
    metric: str
    value: float
    threshold: float
    threshold_basis: str
    reason: str
    context: dict[str, Any] = Field(default_factory=dict)


class AnomalySummary(BaseModel):
    total: int
    by_type: dict[str, int]
    by_severity: dict[str, int]
    reference_time: datetime | None
    thresholds: dict[str, Any]


class AnomalyResponse(BaseModel):
    summary: AnomalySummary
    anomalies: list[Anomaly]


class TicketOut(BaseModel):
    ticket_id: str
    created_at: datetime
    category: str
    priority: str
    status: str
    response_time_hrs: float | None
    resolution_time_hrs: float | None
    agent_id: str
    customer_rating: float | None
    issue_summary: str


class TicketPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[TicketOut]


class StatsResponse(BaseModel):
    total_tickets: int
    by_status: dict[str, int]
    by_category: dict[str, int]
    by_priority: dict[str, int]
    unresolved_tickets: int
    high_or_critical_tickets: int
    avg_customer_rating: float | None
    avg_response_time_hrs: float | None
    avg_resolution_time_hrs: float | None
    median_resolution_time_hrs: float | None
    rated_tickets: int
    resolved_tickets: int
    agents: int
    date_range: dict[str, datetime | None]
    anomaly_count: int


class ErrorResponse(BaseModel):
    error: str
    message: str
    detail: str | None = None
