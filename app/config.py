
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ----- data / storage -------------------------------------------------
    csv_path: Path = Field(default=PROJECT_ROOT / "data" / "support_tickets.csv")
    sqlite_path: Path = Field(default=PROJECT_ROOT / "data" / "tickets.db")
    rebuild_db_on_startup: bool = True

    # ----- LLM ------------------------------------------------------------
    # "groq"       -> Groq free tier (default, recommended)
    # "ollama"     -> local Ollama server
    # "rule_based" -> offline deterministic planner (demo/CI only, see README)
    llm_provider: Literal["groq", "ollama", "rule_based"] = "groq"
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    ollama_model: str = "llama3.1:8b"
    ollama_base_url: str = "http://localhost:11434"
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 1  # one corrective retry on invalid JSON

    # ----- query engine ---------------------------------------------------
    default_row_limit: int = 25
    max_row_limit: int = 200

    # ----- anomaly detection ---------------------------------------------
    # The dataset is historical (Jan-Mar 2024). Using "now" for ticket age
    # would make every open ticket ~2 years old and the rule meaningless, so
    # ageing is measured against a reference time derived from the data.
    # "max_created_at" | "now" | an ISO-8601 timestamp.
    anomaly_reference_time: str = "max_created_at"
    aging_threshold_hours: float = 24.0
    long_resolution_percentile: float = 95.0
    slow_response_percentile: float = 90.0
    low_rating_threshold: int = 2
    min_tickets_for_agent_outlier: int = 5

    # ----- API ------------------------------------------------------------
    api_base_url: str = "http://localhost:8000"  # used by the Streamlit client
    log_level: str = "INFO"

    @field_validator("long_resolution_percentile", "slow_response_percentile")
    @classmethod
    def _valid_percentile(cls, v: float) -> float:
        if not 50.0 <= v < 100.0:
            raise ValueError("percentile must be in [50, 100)")
        return v

    @property
    def llm_configured(self) -> bool:
        if self.llm_provider == "groq":
            return bool(self.groq_api_key)
        return True


@lru_cache
def get_settings() -> Settings:
    return Settings()
