

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from fastapi import Request

from app.config import Settings, get_settings
from app.database.database import TicketDatabase, build_database
from app.query.engine import QueryEngine
from app.services.anomaly_service import AnomalyService
from app.services.llm_service import LLMProvider, LLMQueryPlanner, build_provider
from app.services.query_service import QueryService
from app.utils.data_loader import LoadReport, load_tickets

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    settings: Settings
    db: TicketDatabase
    engine: QueryEngine
    query_service: QueryService
    anomaly_service: AnomalyService
    provider: LLMProvider
    load_report: LoadReport
    tickets_loaded: int
    min_created_at: datetime | None
    max_created_at: datetime | None


def build_state(
    settings: Settings | None = None,
    provider: LLMProvider | None = None,
) -> AppState:
    """Load the CSV, build SQLite, and wire up the services.

    ``provider`` can be injected to swap in a stub LLM for tests.
    """
    settings = settings or get_settings()

    df, report = load_tickets(settings.csv_path)
    if settings.rebuild_db_on_startup or not settings.sqlite_path.exists():
        build_database(df, settings.sqlite_path)

    db = TicketDatabase(settings.sqlite_path)
    tickets_loaded = db.count_tickets()
    min_created_at, max_created_at = db.date_range()

    engine = QueryEngine(db, settings)
    llm_provider = provider or build_provider(settings)
    planner = LLMQueryPlanner(
        llm_provider, settings, min_created_at, max_created_at
    )

    return AppState(
        settings=settings,
        db=db,
        engine=engine,
        query_service=QueryService(planner, engine, settings),
        anomaly_service=AnomalyService(db, settings),
        provider=llm_provider,
        load_report=report,
        tickets_loaded=tickets_loaded,
        min_created_at=min_created_at,
        max_created_at=max_created_at,
    )


def get_state(request: Request) -> AppState:
    state: AppState | None = getattr(request.app.state, "app_state", None)
    if state is None:  # pragma: no cover - only if startup failed
        from app.utils.errors import DatabaseError

        raise DatabaseError(
            "Application is not initialised",
            "Startup did not complete; check the server logs.",
        )
    return state
