"""FastAPI application entrypoint.

"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import routes_anomaly, routes_health, routes_query, routes_tickets
from app.config import get_settings
from app.dependencies import build_state
from app.schemas.response import ErrorResponse
from app.utils.errors import AppError

logger = logging.getLogger(__name__)

# DESCRIPTION = """
# Customer Support Ticket Intelligence System.

# Natural-language questions are translated by an LLM into a **validated
# structured query plan**, which a deterministic SQL engine executes against
# SQLite. The model never computes a statistic and never writes SQL, so every
# number returned is reproducible from the dataset.

# Anomaly detection is entirely deterministic and reports the threshold and
# reference time behind every flag.
# """


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("Starting up: loading %s", settings.csv_path)
    try:
        app.state.app_state = build_state(settings)
        logger.info(
            "Ready: %d tickets, LLM provider=%s configured=%s",
            app.state.app_state.tickets_loaded,
            app.state.app_state.provider.name,
            settings.llm_configured,
        )
        if not settings.llm_configured:
            logger.warning(
                "LLM provider '%s' is not configured; /query will return 503 "
                "until GROQ_API_KEY is set (see .env.example)",
                settings.llm_provider,
            )
    except AppError as exc:
        # Startup failures are fatal for /query but we still want the process
        # up so /health can explain what is wrong.
        app.state.app_state = None
        app.state.startup_error = exc
        logger.error("Startup failed: %s (%s)", exc.message, exc.detail)
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Support Ticket Intelligence API",
    description=DESCRIPTION,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local prototype; tighten before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_health.router)
app.include_router(routes_query.router)
app.include_router(routes_anomaly.router)
app.include_router(routes_tickets.router)


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    logger.warning(
        "%s %s -> %s: %s", request.method, request.url.path, exc.error_code, exc.message
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.error_code, message=exc.message, detail=exc.detail
        ).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="invalid_request",
            message="The request body or query parameters are invalid",
            detail="; ".join(
                f"{'.'.join(str(p) for p in e['loc'][1:])}: {e['msg']}"
                for e in exc.errors()
            )[:500],
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_error",
            message="An unexpected error occurred",
            detail=str(exc)[:300],
        ).model_dump(),
    )


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {
        "name": "Support Ticket Intelligence API",
        "docs": "/docs",
        "health": "/health",
    }
