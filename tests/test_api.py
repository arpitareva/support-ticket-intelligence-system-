"""HTTP contract: status codes, payload shapes, error handling."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.utils.errors import LLMConfigurationError, LLMUnavailableError
from tests.conftest import StubProvider


def seed(client: TestClient, *responses) -> StubProvider:
    """Queue canned LLM responses on the app's stub provider."""
    provider: StubProvider = client.app.state.app_state.provider
    provider.responses.extend(responses)
    return provider


# -- health -----------------------------------------------------------------


def test_health_reports_loaded_tickets(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["tickets_loaded"] == 10
    assert body["database"] == "connected"
    assert body["llm_provider"] == "stub"
    assert body["dataset_range"]["start"].startswith("2024-01-01")


def test_health_is_degraded_without_llm_credentials(client: TestClient) -> None:
    client.app.state.app_state.settings.groq_api_key = None

    body = client.get("/health").json()

    assert body["status"] == "degraded"
    assert body["llm_configured"] is False
    assert body["tickets_loaded"] == 10


def test_root_points_at_the_docs(client: TestClient) -> None:
    assert client.get("/").json()["docs"] == "/docs"


# -- query ------------------------------------------------------------------


def test_query_returns_answer_plan_and_metadata(client: TestClient) -> None:
    seed(
        client,
        {"intent": "count", "filters": [{"field": "status", "op": "eq", "value": "Open"}]},
    )

    response = client.post("/query", json={"question": "How many tickets are open?"})

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "How many tickets are open?"
    assert "2 tickets" in body["answer"]
    assert body["query_plan"]["intent"] == "count"
    assert "status is Open" in body["plan_explanation"]
    assert body["execution"]["llm_provider"] == "stub"
    assert "SELECT COUNT(*)" in body["execution"]["sql"]


def test_query_returns_rows_for_a_list_intent(client: TestClient) -> None:
    seed(
        client,
        {
            "intent": "list",
            "filters": [{"field": "priority", "op": "eq", "value": "Critical"}],
            "limit": 5,
        },
    )

    body = client.post("/query", json={"question": "Show Critical tickets"}).json()

    assert body["result_type"] == "table"
    assert len(body["results"]) == 2
    assert {row["ticket_id"] for row in body["results"]} == {"TKT-002", "TKT-006"}


def test_query_empty_result_is_a_200_with_a_clear_answer(client: TestClient) -> None:
    seed(
        client,
        {
            "intent": "list",
            "filters": [{"field": "agent_id", "op": "eq", "value": "AGT-99"}],
        },
    )

    response = client.post("/query", json={"question": "Show tickets for AGT-99"})

    assert response.status_code == 200
    assert response.json()["results"] == []
    assert "No tickets" in response.json()["answer"]


def test_unsupported_question_returns_422_with_a_reason(client: TestClient) -> None:
    seed(
        client,
        {"intent": "unsupported", "reason": "The dataset has no revenue column."},
    )

    response = client.post("/query", json={"question": "What was our revenue?"})

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "unsupported_question"
    assert "revenue column" in body["detail"]


def test_llm_failure_returns_502_and_no_fabricated_answer(client: TestClient) -> None:
    seed(client, LLMUnavailableError("The language model timed out", "read timeout"))

    response = client.post("/query", json={"question": "How many tickets are open?"})

    assert response.status_code == 502
    body = response.json()
    assert body["error"] == "llm_unavailable"
    assert "answer" not in body


def test_invalid_llm_json_returns_502(client: TestClient) -> None:
    seed(client, "I'm not going to answer that", "still not json")

    response = client.post("/query", json={"question": "How many tickets are open?"})

    assert response.status_code == 502
    assert response.json()["error"] == "llm_invalid_output"


def test_missing_credentials_returns_503(client: TestClient) -> None:
    seed(client, LLMConfigurationError("GROQ_API_KEY is not set"))

    response = client.post("/query", json={"question": "How many tickets are open?"})

    assert response.status_code == 503
    assert response.json()["error"] == "llm_not_configured"


def test_malformed_request_body_returns_422(client: TestClient) -> None:
    assert client.post("/query", json={}).status_code == 422
    assert client.post("/query", json={"question": ""}).status_code == 422
    assert client.post("/query", json={"question": "  "}).status_code == 422
    assert client.post("/query", json={"question": "x" * 5000}).status_code == 422
    assert client.post("/query", json={"question": "ok?", "sql": "DROP"}).status_code == 422


def test_validation_error_payload_is_structured(client: TestClient) -> None:
    body = client.post("/query", json={}).json()

    assert body["error"] == "invalid_request"
    assert "question" in body["detail"]


# -- anomalies --------------------------------------------------------------


def test_anomalies_endpoint_returns_summary_and_items(client: TestClient) -> None:
    response = client.get("/anomalies")

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["total"] == len(body["anomalies"])
    assert body["summary"]["reference_time"].startswith("2024-01-08")
    assert "aging_threshold_hours" in body["summary"]["thresholds"]
    first = body["anomalies"][0]
    for key in ("anomaly_type", "severity", "value", "threshold", "reason"):
        assert key in first


def test_anomalies_type_filter(client: TestClient) -> None:
    body = client.get("/anomalies", params={"anomaly_type": "low_customer_rating"}).json()

    assert set(body["summary"]["by_type"]) == {"low_customer_rating"}
    assert body["summary"]["total"] == 2


def test_anomalies_severity_filter_recomputes_the_summary(client: TestClient) -> None:
    body = client.get("/anomalies", params={"severity": "critical"}).json()

    assert set(body["summary"]["by_severity"]) == {"critical"}
    assert body["summary"]["total"] == len(body["anomalies"])


def test_anomalies_rejects_an_invalid_limit(client: TestClient) -> None:
    assert client.get("/anomalies", params={"limit": 0}).status_code == 422
    assert client.get("/anomalies", params={"limit": 99999}).status_code == 422


# -- tickets and stats ------------------------------------------------------


def test_tickets_pagination(client: TestClient) -> None:
    first = client.get("/tickets", params={"limit": 3, "offset": 0}).json()
    second = client.get("/tickets", params={"limit": 3, "offset": 3}).json()

    assert first["total"] == 10
    assert len(first["items"]) == 3
    assert {t["ticket_id"] for t in first["items"]}.isdisjoint(
        {t["ticket_id"] for t in second["items"]}
    )


def test_tickets_filtering(client: TestClient) -> None:
    body = client.get(
        "/tickets", params={"priority": "Critical", "resolved": False}
    ).json()

    assert body["total"] == 1
    assert body["items"][0]["ticket_id"] == "TKT-002"


def test_tickets_rejects_an_unknown_filter_value(client: TestClient) -> None:
    response = client.get("/tickets", params={"priority": "Hardware"})

    assert response.status_code == 422
    assert response.json()["error"] == "invalid_query_plan"


def test_tickets_limit_is_capped(client: TestClient) -> None:
    assert client.get("/tickets", params={"limit": 5000}).status_code == 422


def test_stats_endpoint(client: TestClient) -> None:
    body = client.get("/stats").json()

    assert body["total_tickets"] == 10
    assert body["by_status"] == {"Resolved": 6, "Open": 2, "Escalated": 2}
    assert body["unresolved_tickets"] == 4
    assert body["high_or_critical_tickets"] == 5
    assert body["rated_tickets"] == 6
    assert body["avg_customer_rating"] == 3.17
    assert body["agents"] == 3
    assert body["anomaly_count"] > 0


def test_openapi_schema_is_served(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    for path in ("/health", "/query", "/anomalies", "/tickets", "/stats"):
        assert path in schema["paths"]
