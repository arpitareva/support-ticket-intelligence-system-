"""Shared fixtures.

Every fixture builds the full stack against a **temporary** CSV and SQLite
file, so tests never touch the real dataset or require a network call. The
LLM is replaced by ``StubProvider``, which returns canned JSON - that is what
makes the test suite deterministic and free to run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.dependencies import AppState, build_state
from app.services.llm_service import LLMProvider

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_CSV = PROJECT_ROOT / "data" / "support_tickets.csv"

# A small, hand-checked fixture dataset. Every expected value in the tests is
# derived from these rows by hand, so a regression in the engine cannot hide
# behind a recomputed expectation.
#
# Summary: 10 tickets, 6 resolved, 4 unresolved (2 Open, 2 Escalated).
#   AGT-01: 3 resolved, ratings 5,4,3 -> avg 4.0
#   AGT-02: 2 resolved, ratings 2,1   -> avg 1.5
#   AGT-03: 1 resolved, rating 4      -> avg 4.0
SAMPLE_CSV = """ticket_id,created_at,category,priority,status,response_time_hrs,resolution_time_hrs,agent_id,customer_rating,issue_summary
TKT-001,2024-01-01 09:00,Billing,High,Resolved,0.5,2.0,AGT-01,5,Incorrect charge on invoice
TKT-002,2024-01-01 10:00,Technical,Critical,Escalated,1.2,,AGT-02,,Login failure after update
TKT-003,2024-01-02 08:00,General,Low,Resolved,3.0,5.0,AGT-01,4,Request for product docs
TKT-004,2024-01-02 14:00,Technical,High,Resolved,0.8,4.0,AGT-01,3,API timeout in production
TKT-005,2024-01-03 10:00,Billing,Medium,Open,2.0,,AGT-03,,Refund not processed
TKT-006,2024-01-04 11:00,Technical,Critical,Resolved,1.0,90.0,AGT-02,2,Data sync not working
TKT-007,2024-01-05 12:00,Billing,Low,Resolved,4.0,3.0,AGT-02,1,Double billing issue
TKT-008,2024-01-06 13:00,General,Medium,Open,2.5,,AGT-03,,How to export data to CSV
TKT-009,2024-01-07 15:00,Technical,High,Escalated,3.5,,AGT-01,,Dashboard not loading
TKT-010,2024-01-08 16:00,General,Low,Resolved,1.5,6.0,AGT-03,4,Feature request: dark mode
"""


class StubProvider(LLMProvider):
    """Returns pre-seeded responses instead of calling a model.

    ``responses`` may hold dicts (serialised to JSON), raw strings (to
    simulate malformed output) or exceptions (to simulate provider failure).
    Responses are consumed in order, which lets a test assert on the
    invalid-output retry path.
    """

    name = "stub"
    model = "stub-model"

    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[list[dict[str, str]]] = []

    def complete_json(self, system: str, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("StubProvider ran out of canned responses")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, str):
            return item
        return json.dumps(item)


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    path = tmp_path / "support_tickets.csv"
    path.write_text(SAMPLE_CSV, encoding="utf-8")
    return path


@pytest.fixture
def settings(sample_csv: Path, tmp_path: Path) -> Settings:
    return Settings(
        csv_path=sample_csv,
        sqlite_path=tmp_path / "tickets.db",
        llm_provider="groq",
        groq_api_key="test-key-not-used",
        rebuild_db_on_startup=True,
        anomaly_reference_time="max_created_at",
        min_tickets_for_agent_outlier=1,
        _env_file=None,
    )


@pytest.fixture
def stub() -> StubProvider:
    return StubProvider()


@pytest.fixture
def state(settings: Settings, stub: StubProvider) -> AppState:
    return build_state(settings, provider=stub)


@pytest.fixture
def client(settings: Settings, stub: StubProvider) -> Iterator[TestClient]:
    """TestClient whose app state is built from the fixture dataset."""
    from app.main import app

    app.state.app_state = build_state(settings, provider=stub)
    with TestClient(app) as test_client:
        # The lifespan handler rebuilds state from the real settings; put the
        # fixture state back so tests run against the small dataset.
        app.state.app_state = build_state(settings, provider=stub)
        yield test_client


@pytest.fixture(scope="session")
def real_settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    """Settings pointed at the shipped 500-row dataset."""
    return Settings(
        csv_path=REAL_CSV,
        sqlite_path=tmp_path_factory.mktemp("real") / "tickets.db",
        llm_provider="rule_based",
        rebuild_db_on_startup=True,
        _env_file=None,
    )
