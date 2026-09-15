"""Anomaly detection endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query

from app.dependencies import AppState, get_state
from app.schemas.response import AnomalyResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["anomalies"])

KNOWN_TYPES = [
    "unresolved_high_priority_aging",
    "long_resolution",
    "slow_first_response_high_priority",
    "low_customer_rating",
    "agent_rating_outlier",
    "agent_resolution_outlier",
    "data_quality_inconsistent_timings",
]


@router.get(
    "/anomalies",
    response_model=AnomalyResponse,
    summary="Detected anomalies with thresholds and reasoning",
)
def anomalies(
    anomaly_type: list[str] | None = Query(
        default=None,
        description=f"Filter by anomaly type. Known types: {', '.join(KNOWN_TYPES)}",
    ),
    severity: list[str] | None = Query(
        default=None, description="Filter by severity: low, medium, high, critical"
    ),
    limit: int = Query(default=200, ge=1, le=1000),
    state: AppState = Depends(get_state),
) -> AnomalyResponse:
    """Run every detection rule and return the flags.

    Detection is fully deterministic; the summary block reports the exact
    thresholds and reference time used, so results are reproducible.
    """
    response = state.anomaly_service.detect(types=anomaly_type)
    if severity:
        wanted = {s.lower() for s in severity}
        filtered = [a for a in response.anomalies if a.severity in wanted]
        response.anomalies = filtered
        response.summary.total = len(filtered)
        response.summary.by_type = _tally(filtered, "anomaly_type")
        response.summary.by_severity = _tally(filtered, "severity")
    response.anomalies = response.anomalies[:limit]
    return response


def _tally(items: list, attribute: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        key = getattr(item, attribute)
        out[key] = out.get(key, 0) + 1
    return out
