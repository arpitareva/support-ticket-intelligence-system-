"""Natural-language query endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Body, Depends

from app.dependencies import AppState, get_state
from app.schemas.query_plan import QueryRequest
from app.schemas.response import ErrorResponse, QueryResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a natural-language question about the tickets",
    responses={
        422: {"model": ErrorResponse, "description": "Question out of scope"},
        502: {"model": ErrorResponse, "description": "LLM unavailable or invalid output"},
        503: {"model": ErrorResponse, "description": "LLM not configured / database down"},
    },
)
def query(
    payload: QueryRequest = Body(
        ...,
        examples=[{"question": "How many Critical tickets are unresolved?"}],
    ),
    state: AppState = Depends(get_state),
) -> QueryResponse:
    """Translate the question into a validated query plan, execute it in SQL,
    and return the exact result together with the plan that produced it.

    Errors are surfaced rather than papered over: if the model cannot produce
    a usable plan, the response is a 502/422 with an explanation, never a
    plausible-sounding number.
    """
    logger.info("POST /query: %r", payload.question[:200])
    return state.query_service.answer(payload.question)
