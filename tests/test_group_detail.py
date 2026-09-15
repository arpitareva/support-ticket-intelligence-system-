"""Two-stage questions: an aggregate picks a group, then its rows are listed.

These cover the class of question that previously fell through to
``intent: unsupported`` - "which agent has the most X, and what are those
ticket IDs?".
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.dependencies import AppState, build_state
from app.schemas.query_plan import (
    Aggregation,
    DetailSpec,
    Field_,
    Filter,
    GroupField,
    Intent,
    Metric,
    Operator,
    QueryPlan,
    Sort,
    SortDirection,
)
from app.services.answer_formatter import describe_plan, format_answer
from app.services.llm_service import RuleBasedProvider
from app.utils.data_loader import load_tickets

TARGET_QUESTION = (
    "Among unresolved High and Critical tickets, which agent currently has "
    "the most tickets, and what are those ticket IDs?"
)


def busiest_agent_plan(**overrides) -> QueryPlan:
    payload = {
        "intent": "group_detail",
        "filters": [
            {"field": "resolved", "op": "eq", "value": False},
            {"field": "priority", "op": "in", "value": ["High", "Critical"]},
        ],
        "group_by": "agent_id",
        "metric": {"agg": "count"},
        "sort": {"by": "metric", "direction": "desc"},
        "limit": 1,
        "detail": {"sort": {"by": "created_at", "direction": "asc"}, "limit": 50},
    }
    payload.update(overrides)
    return QueryPlan.model_validate(payload)


# -- plan validation --------------------------------------------------------


def test_group_detail_requires_group_by() -> None:
    with pytest.raises(ValueError, match="requires group_by"):
        QueryPlan(intent=Intent.GROUP_DETAIL)


def test_group_detail_defaults_to_a_single_winning_group() -> None:
    plan = QueryPlan(intent=Intent.GROUP_DETAIL, group_by=GroupField.AGENT_ID)

    assert plan.limit == 1
    assert plan.detail == DetailSpec()
    assert plan.detail.limit == 50


def test_detail_stage_sort_is_independent_of_group_ranking() -> None:
    plan = busiest_agent_plan(
        sort={"by": "metric", "direction": "desc"},
        detail={"sort": {"by": "response_time_hrs", "direction": "desc"}, "limit": 5},
    )

    assert plan.sort.direction is SortDirection.DESC
    assert plan.detail.sort.by == "response_time_hrs"
    assert plan.detail.limit == 5


def test_detail_spec_rejects_unknown_sort_field() -> None:
    with pytest.raises(ValueError, match="cannot sort by"):
        DetailSpec(sort=Sort(by="customer_email"))


def test_detail_limit_is_bounded() -> None:
    with pytest.raises(ValueError):
        DetailSpec(limit=0)
    with pytest.raises(ValueError):
        DetailSpec(limit=5000)


# -- execution against the fixture dataset ---------------------------------


def test_ranks_group_then_lists_only_that_group_rows(state: AppState) -> None:
    """Unresolved High/Critical in the fixture: AGT-02 (TKT-002) and
    AGT-01 (TKT-009) - one each, so this is a tie."""
    plan = busiest_agent_plan()
    result = state.engine.execute(plan)

    assert result.result_type == "group_detail"
    assert {row["agent_id"] for row in result.group_rows} == {"AGT-01", "AGT-02"}
    assert all(row["ticket_count"] == 1 for row in result.group_rows)
    assert {row["ticket_id"] for row in result.rows} == {"TKT-002", "TKT-009"}


def test_detail_rows_respect_the_stage_one_filters(state: AppState) -> None:
    """The winner's *other* tickets must not leak into the detail list."""
    plan = busiest_agent_plan(
        filters=[{"field": "status", "op": "eq", "value": "Open"}]
    )
    result = state.engine.execute(plan)

    # AGT-03 owns both Open tickets; it also owns resolved TKT-010, which must
    # not appear because the Open filter applies to both stages.
    assert [row["agent_id"] for row in result.group_rows] == ["AGT-03"]
    assert {row["ticket_id"] for row in result.rows} == {"TKT-005", "TKT-008"}
    assert all(row["status"] == "Open" for row in result.rows)


def test_winner_selected_by_an_average_not_a_count(state: AppState) -> None:
    plan = busiest_agent_plan(
        filters=[{"field": "resolved", "op": "eq", "value": True}],
        metric={"agg": "avg", "field": "resolution_time_hrs"},
        detail={"sort": {"by": "resolution_time_hrs", "direction": "desc"}, "limit": 10},
    )
    result = state.engine.execute(plan)

    # AGT-02 resolved 90.0 and 3.0 -> avg 46.5, the slowest agent.
    assert [row["agent_id"] for row in result.group_rows] == ["AGT-02"]
    assert result.group_rows[0]["avg_resolution_time_hrs"] == 46.5
    assert [row["ticket_id"] for row in result.rows] == ["TKT-006", "TKT-007"]


def test_lowest_direction_selects_the_bottom_group(state: AppState) -> None:
    plan = busiest_agent_plan(
        filters=[{"field": "resolved", "op": "eq", "value": True}],
        metric={"agg": "avg", "field": "customer_rating"},
        sort={"by": "metric", "direction": "asc"},
    )
    result = state.engine.execute(plan)

    assert [row["agent_id"] for row in result.group_rows] == ["AGT-02"]
    assert result.group_rows[0]["avg_customer_rating"] == 1.5


def test_grouping_by_category_works_too(state: AppState) -> None:
    plan = busiest_agent_plan(
        filters=[], group_by="category", metric={"agg": "count"}
    )
    result = state.engine.execute(plan)

    assert [row["category"] for row in result.group_rows] == ["Technical"]
    assert len(result.rows) == 4
    assert all(row["category"] == "Technical" for row in result.rows)


def test_having_min_count_applies_to_stage_one(state: AppState) -> None:
    plan = busiest_agent_plan(
        filters=[{"field": "resolved", "op": "eq", "value": True}],
        metric={"agg": "avg", "field": "customer_rating"},
        sort={"by": "metric", "direction": "asc"},
        having_min_count=3,
    )
    result = state.engine.execute(plan)

    # AGT-02 has only 2 resolved tickets, so AGT-01 (3) wins by default.
    assert [row["agent_id"] for row in result.group_rows] == ["AGT-01"]


def test_detail_limit_truncates_and_says_so(state: AppState) -> None:
    plan = busiest_agent_plan(
        filters=[], group_by="category", detail={"limit": 2}
    )
    result = state.engine.execute(plan)

    assert len(result.rows) == 2
    assert result.matched_rows == 4
    assert any("Listing the first 2 of 4" in note for note in result.notes)


def test_no_matching_groups_returns_empty_not_an_error(state: AppState) -> None:
    plan = busiest_agent_plan(
        filters=[{"field": "agent_id", "op": "eq", "value": "AGT-99"}]
    )
    result = state.engine.execute(plan)

    assert result.group_rows == []
    assert result.rows == []
    assert "No agent_id values matched" in format_answer(plan, result)


def test_group_values_are_bound_as_parameters(state: AppState) -> None:
    """Stage two must parameterise the winners, never interpolate them."""
    plan = busiest_agent_plan()
    result = state.engine.execute(plan)

    detail_sql = result.sql[-1]
    assert "agent_id IN (?, ?)" in detail_sql
    assert "AGT-" not in detail_sql


# -- answer text ------------------------------------------------------------


def test_answer_names_the_winner_and_the_ticket_ids(state: AppState) -> None:
    plan = busiest_agent_plan(
        filters=[{"field": "status", "op": "eq", "value": "Open"}]
    )
    answer = format_answer(plan, state.engine.execute(plan))

    assert "AGT-03" in answer
    assert "TKT-005" in answer and "TKT-008" in answer
    assert "highest number of tickets" in answer


def test_answer_reports_a_tie_at_the_top(state: AppState) -> None:
    plan = busiest_agent_plan()
    answer = format_answer(plan, state.engine.execute(plan))

    assert "tied for the highest" in answer
    assert "AGT-01" in answer and "AGT-02" in answer


def test_plan_explanation_mentions_both_stages(state: AppState) -> None:
    explanation = describe_plan(busiest_agent_plan())

    assert "grouped by agent_id" in explanation
    assert "then list matching tickets" in explanation


# -- planner and API --------------------------------------------------------


def test_rule_based_planner_recognises_the_two_part_shape(
    settings: Settings,
) -> None:
    payload = json.loads(
        RuleBasedProvider(settings).complete_json(
            "", [{"role": "user", "content": TARGET_QUESTION.lower()}]
        )
    )

    assert payload["intent"] == "group_detail"
    assert payload["group_by"] == "agent_id"
    assert {"field": "priority", "op": "in", "value": ["Critical", "High"]} in payload[
        "filters"
    ]
    assert {"field": "resolved", "op": "eq", "value": False} in payload["filters"]


@pytest.mark.parametrize(
    "question",
    [
        "which agent has the most open tickets and what are their ticket ids?",
        "which category has the longest average resolution time? show me those tickets.",
    ],
)
def test_rule_based_planner_generalises(settings: Settings, question: str) -> None:
    payload = json.loads(
        RuleBasedProvider(settings).complete_json(
            "", [{"role": "user", "content": question}]
        )
    )

    assert payload["intent"] == "group_detail"


def test_plain_superlative_still_uses_group_aggregate(settings: Settings) -> None:
    """Adding the drill-down must not hijack single-stage questions."""
    payload = json.loads(
        RuleBasedProvider(settings).complete_json(
            "", [{"role": "user", "content": "which agent resolved the most tickets?"}]
        )
    )

    assert payload["intent"] == "group_aggregate"


def test_api_returns_groups_and_results(client: TestClient) -> None:
    provider = client.app.state.app_state.provider
    provider.responses.append(busiest_agent_plan().model_dump(mode="json"))

    response = client.post("/query", json={"question": TARGET_QUESTION})

    assert response.status_code == 200
    body = response.json()
    assert body["result_type"] == "group_detail"
    assert len(body["groups"]) == 2
    assert {row["ticket_id"] for row in body["results"]} == {"TKT-002", "TKT-009"}
    assert "agent_id IN (?, ?)" in body["execution"]["sql"]


def test_system_prompt_teaches_the_new_intent(state: AppState) -> None:
    prompt = state.query_service.planner.system_prompt

    assert "group_detail" in prompt
    assert '"detail"' in prompt


# -- the real dataset -------------------------------------------------------


def test_target_question_on_the_shipped_dataset(real_settings: Settings) -> None:
    """End-to-end check of the reported failure, with hand-verified numbers."""
    state = build_state(real_settings)
    result = state.engine.execute(busiest_agent_plan())

    # Unresolved High/Critical: AGT-07 and AGT-11 have 11 each (verified
    # independently against the CSV), 22 tickets between them.
    assert {row["agent_id"] for row in result.group_rows} == {"AGT-07", "AGT-11"}
    assert all(row["ticket_count"] == 11 for row in result.group_rows)
    assert result.matched_rows == 22
    assert len(result.rows) == 22

    df, _ = load_tickets(real_settings.csv_path)
    expected = set(
        df.loc[
            (df["status"] != "Resolved")
            & df["priority"].isin(["High", "Critical"])
            & df["agent_id"].isin(["AGT-07", "AGT-11"]),
            "ticket_id",
        ]
    )
    assert {row["ticket_id"] for row in result.rows} == expected
