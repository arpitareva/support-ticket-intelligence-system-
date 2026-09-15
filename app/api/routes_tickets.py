"""Direct ticket access and dataset statistics.

These endpoints bypass the LLM entirely: the UI dashboard and any programmatic
consumer should not pay for a model round-trip to fetch a filtered page of
rows or headline counts.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query

from app.dependencies import AppState, get_state
from app.schemas.query_plan import (
    CATEGORIES,
    PRIORITIES,
    STATUSES,
    Field_,
    Filter,
    Operator,
)
from app.schemas.response import StatsResponse, TicketPage
from app.utils.errors import AppError, QueryPlanError

logger = logging.getLogger(__name__)
router = APIRouter(tags=["tickets"])


@router.get("/tickets", response_model=TicketPage, summary="List tickets (paginated)")
def list_tickets(
    category: str | None = Query(default=None, description=f"One of {CATEGORIES}"),
    priority: str | None = Query(default=None, description=f"One of {PRIORITIES}"),
    status: str | None = Query(default=None, description=f"One of {STATUSES}"),
    agent_id: str | None = Query(default=None, max_length=20),
    resolved: bool | None = Query(default=None),
    limit: int = Query(default=25, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    state: AppState = Depends(get_state),
) -> TicketPage:
    """Filter and page through tickets.

    Filters are converted into the same validated ``Filter`` objects the LLM
    path produces, so query-string input gets identical whitelisting.
    """
    filters: list[Filter] = []
    raw = {
        Field_.CATEGORY: category,
        Field_.PRIORITY: priority,
        Field_.STATUS: status,
        Field_.AGENT_ID: agent_id,
    }
    try:
        for field_, value in raw.items():
            if value is not None:
                filters.append(Filter(field=field_, op=Operator.EQ, value=value))
        if resolved is not None:
            filters.append(
                Filter(field=Field_.RESOLVED, op=Operator.EQ, value=resolved)
            )
    except ValueError as exc:
        raise QueryPlanError("Invalid filter value", str(exc)) from exc

    total, rows = state.engine.list_tickets(
        limit=limit, offset=offset, filters=filters
    )
    return TicketPage(total=total, limit=limit, offset=offset, items=rows)


@router.get("/stats", response_model=StatsResponse, summary="Dataset statistics")
def stats(state: AppState = Depends(get_state)) -> StatsResponse:
    """Headline figures for the dashboard, plus the current anomaly count."""
    payload = state.engine.dataset_stats()
    try:
        payload["anomaly_count"] = state.anomaly_service.count()
    except AppError as exc:  # pragma: no cover - defensive
        logger.error("anomaly count failed: %s", exc.message)
        payload["anomaly_count"] = 0
    return StatsResponse(**payload)
