"""Health and readiness."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from app.dependencies import AppState, get_state
from app.schemas.response import HealthResponse
from app.utils.errors import AppError

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])

VERSION = "1.0.0"


@router.get("/health", response_model=HealthResponse, summary="Service health")
def health(state: AppState = Depends(get_state)) -> HealthResponse:
    """Report database reachability, row count and LLM configuration.

    Always returns 200 so an orchestrator can distinguish "the process is up
    but degraded" from "the process is unreachable". The ``status`` field
    carries the verdict: ``healthy``, ``degraded`` (database fine, LLM not
    configured - deterministic endpoints still work) or ``unhealthy``.
    """
    database = "connected"
    tickets = state.tickets_loaded
    try:
        tickets = state.db.count_tickets()
    except AppError as exc:
        database = f"error: {exc.message}"
        logger.error("health check: database unreachable: %s", exc.detail)

    reference_time = None
    if database == "connected":
        try:
            df = state.db.load_dataframe()
            reference_time, _ = state.anomaly_service.resolve_reference_time(df)
        except AppError:
            reference_time = None

    if database != "connected" or tickets == 0:
        status = "unhealthy"
    elif not state.settings.llm_configured:
        status = "degraded"
    else:
        status = "healthy"

    return HealthResponse(
        status=status,
        database=database,
        tickets_loaded=tickets,
        llm_provider=state.provider.name,
        llm_configured=state.settings.llm_configured,
        dataset_range={"start": state.min_created_at, "end": state.max_created_at},
        anomaly_reference_time=reference_time,
        version=VERSION,
    )
