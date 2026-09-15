"""Domain-level exceptions.

Each exception maps to a single HTTP status code in ``app/main.py`` so the
API layer never has to guess what went wrong deeper in the stack.
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for all expected application failures."""

    status_code = 500
    error_code = "internal_error"

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class DataLoadError(AppError):
    """CSV missing, unreadable, empty, or schema/dtype validation failed."""

    status_code = 500
    error_code = "data_load_error"


class DatabaseError(AppError):
    """SQLite is unreachable or the tickets table is missing/empty."""

    status_code = 503
    error_code = "database_error"


class LLMConfigurationError(AppError):
    """Provider selected but not configured (e.g. missing GROQ_API_KEY)."""

    status_code = 503
    error_code = "llm_not_configured"


class LLMUnavailableError(AppError):
    """Network failure, timeout, rate limit, or non-2xx from the provider."""

    status_code = 502
    error_code = "llm_unavailable"


class LLMOutputError(AppError):
    """The LLM replied, but its output was not a valid query plan."""

    status_code = 502
    error_code = "llm_invalid_output"


class UnsupportedQuestionError(AppError):
    """The question is out of scope for the dataset / query engine."""

    status_code = 422
    error_code = "unsupported_question"


class QueryPlanError(AppError):
    """A structurally valid plan that the engine cannot execute."""

    status_code = 422
    error_code = "invalid_query_plan"
