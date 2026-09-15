"""LLM integration: JSON extraction, validation, repair retry, failures.

No test here touches the network. Provider behaviour is simulated with
``StubProvider`` and with monkeypatched ``httpx`` calls.
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.config import Settings
from app.services.llm_service import (
    GroqProvider,
    LLMQueryPlanner,
    OllamaProvider,
    RuleBasedProvider,
    build_provider,
    extract_json,
)
from app.schemas.query_plan import Aggregation, Field_, Intent
from app.utils.errors import (
    LLMConfigurationError,
    LLMOutputError,
    LLMUnavailableError,
)
from tests.conftest import StubProvider

COUNT_PLAN = {
    "intent": "count",
    "filters": [{"field": "status", "op": "eq", "value": "Open"}],
}


def planner(settings: Settings, responses: list) -> tuple[LLMQueryPlanner, StubProvider]:
    stub = StubProvider(responses)
    return LLMQueryPlanner(stub, settings), stub


# -- JSON extraction --------------------------------------------------------


def test_extract_plain_json() -> None:
    assert extract_json('{"intent": "count"}') == {"intent": "count"}


def test_extract_json_from_markdown_fence() -> None:
    raw = '```json\n{"intent": "count"}\n```'
    assert extract_json(raw) == {"intent": "count"}


def test_extract_json_surrounded_by_prose() -> None:
    raw = 'Sure! Here is the plan:\n{"intent": "count"}\nHope that helps.'
    assert extract_json(raw) == {"intent": "count"}


def test_extract_rejects_non_json() -> None:
    with pytest.raises(LLMOutputError, match="did not return JSON"):
        extract_json("I cannot help with that.")


def test_extract_rejects_a_json_array() -> None:
    with pytest.raises(LLMOutputError, match="not an object"):
        extract_json("[1, 2, 3]")


# -- planning ---------------------------------------------------------------


def test_valid_plan_is_returned(settings: Settings) -> None:
    query_planner, stub = planner(settings, [COUNT_PLAN])

    plan, meta = query_planner.plan("How many tickets are open?")

    assert plan.intent is Intent.COUNT
    assert meta["llm_repair_attempts"] == 0
    assert len(stub.calls) == 1


def test_null_valued_keys_are_tolerated(settings: Settings) -> None:
    """Models routinely emit explicit nulls for unused keys."""
    query_planner, _ = planner(
        settings,
        [
            {
                "intent": "aggregate",
                "filters": [],
                "group_by": None,
                "metric": {"agg": "avg", "field": "customer_rating"},
                "limit": None,
                "having_min_count": None,
            }
        ],
    )

    plan, _ = query_planner.plan("What is the average rating?")

    assert plan.metric.agg is Aggregation.AVG
    assert plan.metric.field is Field_.CUSTOMER_RATING


def test_single_key_envelope_is_unwrapped(settings: Settings) -> None:
    query_planner, _ = planner(settings, [{"query_plan": COUNT_PLAN}])

    plan, _ = query_planner.plan("How many tickets are open?")

    assert plan.intent is Intent.COUNT


def test_invalid_json_triggers_one_repair_attempt(settings: Settings) -> None:
    query_planner, stub = planner(settings, ["not json at all", COUNT_PLAN])

    plan, meta = query_planner.plan("How many tickets are open?")

    assert plan.intent is Intent.COUNT
    assert meta["llm_repair_attempts"] == 1
    assert len(stub.calls) == 2
    # The retry must include the validator's complaint.
    assert "rejected by the schema validator" in stub.calls[1][-1]["content"]


def test_persistent_invalid_json_raises(settings: Settings) -> None:
    query_planner, _ = planner(settings, ["nope", "still nope"])

    with pytest.raises(LLMOutputError, match="valid query plan"):
        query_planner.plan("How many tickets are open?")


def test_schema_violation_is_rejected_not_executed(settings: Settings) -> None:
    """A hallucinated column must never reach the query engine."""
    bad = {"intent": "count", "filters": [{"field": "customer_email", "op": "eq", "value": "x"}]}
    query_planner, _ = planner(settings, [bad, bad])

    with pytest.raises(LLMOutputError):
        query_planner.plan("How many tickets has bob@example.com filed?")


def test_generated_sql_is_never_accepted(settings: Settings) -> None:
    raw = json.dumps({"intent": "count", "sql": "DROP TABLE tickets"})
    query_planner, _ = planner(settings, [raw, raw])

    with pytest.raises(LLMOutputError):
        query_planner.plan("delete everything")


def test_provider_failure_propagates(settings: Settings) -> None:
    query_planner, _ = planner(
        settings, [LLMUnavailableError("The language model timed out")]
    )

    with pytest.raises(LLMUnavailableError):
        query_planner.plan("How many tickets are open?")


def test_unsupported_intent_is_a_valid_plan(settings: Settings) -> None:
    """Refusing to answer is a first-class outcome, not an error."""
    query_planner, _ = planner(
        settings, [{"intent": "unsupported", "reason": "No churn data exists."}]
    )

    plan, _ = query_planner.plan("Who will churn next quarter?")

    assert plan.intent is Intent.UNSUPPORTED
    assert plan.reason == "No churn data exists."


def test_system_prompt_carries_the_dataset_vocabulary(settings: Settings) -> None:
    from datetime import datetime

    query_planner = LLMQueryPlanner(
        StubProvider(), settings, datetime(2024, 1, 1), datetime(2024, 3, 30)
    )

    prompt = query_planner.system_prompt

    assert "Billing | Technical | General" in prompt
    assert "2024-03-30" in prompt
    assert "resolution_time_hrs" in prompt


# -- provider selection and configuration ----------------------------------


def test_build_provider_honours_configuration(settings: Settings) -> None:
    settings.llm_provider = "groq"
    assert isinstance(build_provider(settings), GroqProvider)
    settings.llm_provider = "ollama"
    assert isinstance(build_provider(settings), OllamaProvider)
    settings.llm_provider = "rule_based"
    assert isinstance(build_provider(settings), RuleBasedProvider)


def test_missing_api_key_is_reported_clearly(settings: Settings) -> None:
    settings.groq_api_key = None
    provider = GroqProvider(settings)

    with pytest.raises(LLMConfigurationError, match="GROQ_API_KEY"):
        provider.complete_json("system", [{"role": "user", "content": "hi"}])


def test_llm_configured_flag(settings: Settings) -> None:
    settings.llm_provider = "groq"
    settings.groq_api_key = None
    assert settings.llm_configured is False
    settings.groq_api_key = "key"
    assert settings.llm_configured is True


def test_groq_rate_limit_maps_to_unavailable(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_post(*args, **kwargs):
        return httpx.Response(429, json={"error": "rate limit"})

    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(LLMUnavailableError, match="rate limit"):
        GroqProvider(settings).complete_json("system", [{"role": "user", "content": "x"}])


def test_groq_bad_key_maps_to_configuration_error(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: httpx.Response(401, json={"error": "bad key"})
    )

    with pytest.raises(LLMConfigurationError):
        GroqProvider(settings).complete_json("system", [{"role": "user", "content": "x"}])


def test_groq_timeout_maps_to_unavailable(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def raise_timeout(*args, **kwargs):
        raise httpx.TimeoutException("too slow")

    monkeypatch.setattr(httpx, "post", raise_timeout)

    with pytest.raises(LLMUnavailableError, match="timed out"):
        GroqProvider(settings).complete_json("system", [{"role": "user", "content": "x"}])


def test_groq_sends_json_mode_and_zero_temperature(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update({"url": url, "json": json, "headers": headers})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"intent":"count"}'}}]},
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    GroqProvider(settings).complete_json("system", [{"role": "user", "content": "x"}])

    assert captured["json"]["temperature"] == 0
    assert captured["json"]["response_format"] == {"type": "json_object"}
    assert captured["headers"]["Authorization"].startswith("Bearer ")


def test_ollama_unreachable_gives_actionable_message(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def raise_connect(*args, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", raise_connect)

    with pytest.raises(LLMUnavailableError) as exc_info:
        OllamaProvider(settings).complete_json("system", [{"role": "user", "content": "x"}])

    assert "ollama serve" in (exc_info.value.detail or "")


# -- offline rule-based planner --------------------------------------------


@pytest.mark.parametrize(
    "question,expected_intent",
    [
        ("How many tickets are currently open?", "count"),
        ("What is the average customer rating?", "aggregate"),
        ("Which agent resolved the most tickets?", "group_aggregate"),
        ("Show me all Critical tickets not resolved within 12 hours.", "list"),
        ("Write me a poem about support tickets", "unsupported"),
    ],
)
def test_rule_based_planner_covers_common_shapes(
    settings: Settings, question: str, expected_intent: str
) -> None:
    provider = RuleBasedProvider(settings)

    payload = json.loads(provider.complete_json("", [{"role": "user", "content": question}]))

    assert payload["intent"] == expected_intent


def test_rule_based_planner_reports_its_own_limits(settings: Settings) -> None:
    provider = RuleBasedProvider(settings)

    payload = json.loads(
        provider.complete_json("", [{"role": "user", "content": "explain quantum physics"}])
    )

    assert "GROQ_API_KEY" in payload["reason"]


def test_rule_based_planner_handles_relative_comparisons(settings: Settings) -> None:
    """A "below-average X and above-average Y" question must not be answered
    as a plain aggregate, which would be confidently wrong."""
    provider = RuleBasedProvider(settings)

    payload = json.loads(
        provider.complete_json(
            "",
            [
                {
                    "role": "user",
                    "content": "which agents have both a below-average rating "
                    "and above-average resolution time?",
                }
            ],
        )
    )

    assert payload["intent"] == "relative_outliers"
    assert payload["group_by"] == "agent_id"
    directions = {c["direction"] for c in payload["relative_conditions"]}
    fields = {c["metric"]["field"] for c in payload["relative_conditions"]}
    assert directions == {"below", "above"}
    assert fields == {"customer_rating", "resolution_time_hrs"}
