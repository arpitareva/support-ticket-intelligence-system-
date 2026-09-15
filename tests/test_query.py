"""Query engine correctness.

Expected values are hand-derived from the fixture dataset in ``conftest.py``,
not recomputed with pandas, so these tests genuinely pin the behaviour.
"""

from __future__ import annotations

import pytest

from app.dependencies import AppState
from app.schemas.query_plan import (
    Aggregation,
    CompareArm,
    Direction,
    Field_,
    Filter,
    GroupField,
    Intent,
    Metric,
    Operator,
    QueryPlan,
    RelativeCondition,
    Sort,
    SortDirection,
    TimeRange,
)
from app.services.answer_formatter import format_answer
from app.utils.errors import QueryPlanError


def run(state: AppState, plan: QueryPlan):
    return state.engine.execute(plan)


# -- counting ---------------------------------------------------------------


def test_count_all_tickets(state: AppState) -> None:
    result = run(state, QueryPlan(intent=Intent.COUNT))
    assert result.scalar == 10


def test_count_with_status_filter(state: AppState) -> None:
    plan = QueryPlan(
        intent=Intent.COUNT,
        filters=[Filter(field=Field_.STATUS, op=Operator.EQ, value="Open")],
    )
    assert run(state, plan).scalar == 2


def test_count_unresolved_uses_virtual_resolved_field(state: AppState) -> None:
    plan = QueryPlan(
        intent=Intent.COUNT,
        filters=[Filter(field=Field_.RESOLVED, op=Operator.EQ, value=False)],
    )
    # 2 Open + 2 Escalated
    assert run(state, plan).scalar == 4


def test_count_with_two_filters(state: AppState) -> None:
    plan = QueryPlan(
        intent=Intent.COUNT,
        filters=[
            Filter(field=Field_.PRIORITY, op=Operator.EQ, value="Critical"),
            Filter(field=Field_.RESOLVED, op=Operator.EQ, value=False),
        ],
    )
    assert run(state, plan).scalar == 1  # TKT-002


def test_count_with_in_operator(state: AppState) -> None:
    plan = QueryPlan(
        intent=Intent.COUNT,
        filters=[
            Filter(
                field=Field_.PRIORITY, op=Operator.IN, value=["High", "Critical"]
            )
        ],
    )
    assert run(state, plan).scalar == 5


def test_count_empty_result_is_zero_not_an_error(state: AppState) -> None:
    plan = QueryPlan(
        intent=Intent.COUNT,
        filters=[
            Filter(field=Field_.CATEGORY, op=Operator.EQ, value="Billing"),
            Filter(field=Field_.PRIORITY, op=Operator.EQ, value="Critical"),
        ],
    )
    result = run(state, plan)
    assert result.scalar == 0
    assert "There are 0 tickets" in format_answer(plan, result)


# -- averages and aggregates ------------------------------------------------


def test_average_customer_rating_ignores_nulls(state: AppState) -> None:
    plan = QueryPlan(
        intent=Intent.AGGREGATE,
        metric=Metric(agg=Aggregation.AVG, field=Field_.CUSTOMER_RATING),
    )
    result = run(state, plan)
    # ratings 5,4,3,2,1,4 over 6 rated tickets -> 19/6 = 3.17
    assert result.scalar == 3.17
    assert result.sample_size == 6
    assert result.matched_rows == 10
    assert any("have no customer_rating" in note for note in result.notes)


def test_average_for_a_category(state: AppState) -> None:
    plan = QueryPlan(
        intent=Intent.AGGREGATE,
        filters=[Filter(field=Field_.CATEGORY, op=Operator.EQ, value="Technical")],
        metric=Metric(agg=Aggregation.AVG, field=Field_.CUSTOMER_RATING),
    )
    # Technical rated tickets: TKT-004 (3), TKT-006 (2) -> 2.5
    assert run(state, plan).scalar == 2.5


def test_max_and_min_resolution_time(state: AppState) -> None:
    for agg, expected in ((Aggregation.MAX, 90.0), (Aggregation.MIN, 2.0)):
        plan = QueryPlan(
            intent=Intent.AGGREGATE,
            metric=Metric(agg=agg, field=Field_.RESOLUTION_TIME_HRS),
        )
        assert run(state, plan).scalar == expected


def test_median_computed_outside_sql(state: AppState) -> None:
    plan = QueryPlan(
        intent=Intent.AGGREGATE,
        metric=Metric(agg=Aggregation.MEDIAN, field=Field_.RESOLUTION_TIME_HRS),
    )
    # resolved times sorted: 2,3,4,5,6,90 -> median (4+5)/2 = 4.5
    assert run(state, plan).scalar == 4.5


def test_aggregate_with_no_values_returns_none_not_zero(state: AppState) -> None:
    plan = QueryPlan(
        intent=Intent.AGGREGATE,
        filters=[Filter(field=Field_.RESOLVED, op=Operator.EQ, value=False)],
        metric=Metric(agg=Aggregation.AVG, field=Field_.CUSTOMER_RATING),
    )
    result = run(state, plan)
    assert result.scalar is None
    assert "No data is available" in format_answer(plan, result)


def test_case_insensitive_filter_values_are_normalised(state: AppState) -> None:
    plan = QueryPlan(
        intent=Intent.COUNT,
        filters=[Filter(field=Field_.CATEGORY, op=Operator.EQ, value="billing")],
    )
    assert run(state, plan).scalar == 3


# -- listing ----------------------------------------------------------------


def test_list_applies_limit_and_reports_total(state: AppState) -> None:
    plan = QueryPlan(
        intent=Intent.LIST,
        filters=[Filter(field=Field_.CATEGORY, op=Operator.EQ, value="Technical")],
        limit=2,
        sort=Sort(by="ticket_id", direction=SortDirection.ASC),
    )
    result = run(state, plan)
    assert [row["ticket_id"] for row in result.rows] == ["TKT-002", "TKT-004"]
    assert result.matched_rows == 4
    assert any("Showing the first 2" in note for note in result.notes)


def test_list_empty_result(state: AppState) -> None:
    plan = QueryPlan(
        intent=Intent.LIST,
        filters=[Filter(field=Field_.AGENT_ID, op=Operator.EQ, value="AGT-99")],
    )
    result = run(state, plan)
    assert result.rows == []
    assert "No tickets" in format_answer(plan, result)


def test_or_block_covers_sla_breach(state: AppState) -> None:
    """"Critical tickets not resolved within 12 hours" must include open ones."""
    plan = QueryPlan(
        intent=Intent.LIST,
        filters=[Filter(field=Field_.PRIORITY, op=Operator.EQ, value="Critical")],
        any_of=[
            Filter(field=Field_.RESOLUTION_TIME_HRS, op=Operator.GT, value=12),
            Filter(field=Field_.RESOLVED, op=Operator.EQ, value=False),
        ],
    )
    result = run(state, plan)
    # TKT-006 resolved in 90h, TKT-002 still escalated
    assert {row["ticket_id"] for row in result.rows} == {"TKT-002", "TKT-006"}


def test_single_element_or_block_is_normalised_to_and(state: AppState) -> None:
    plan = QueryPlan(
        intent=Intent.COUNT,
        any_of=[Filter(field=Field_.STATUS, op=Operator.EQ, value="Open")],
    )
    assert plan.any_of == []
    assert len(plan.filters) == 1


def test_time_range_filter(state: AppState) -> None:
    plan = QueryPlan(
        intent=Intent.COUNT,
        time_range=TimeRange(start="2024-01-01 00:00", end="2024-01-02 23:59"),
    )
    assert run(state, plan).scalar == 4


# -- grouping and agent performance ----------------------------------------


def test_agent_with_most_resolved_tickets(state: AppState) -> None:
    plan = QueryPlan(
        intent=Intent.GROUP_AGGREGATE,
        filters=[Filter(field=Field_.RESOLVED, op=Operator.EQ, value=True)],
        group_by=GroupField.AGENT_ID,
        metric=Metric(agg=Aggregation.COUNT),
        sort=Sort(by="metric", direction=SortDirection.DESC),
        limit=1,
    )
    result = run(state, plan)
    assert result.rows[0]["agent_id"] == "AGT-01"
    assert result.rows[0]["ticket_count"] == 3


def test_lowest_average_rating_by_agent(state: AppState) -> None:
    plan = QueryPlan(
        intent=Intent.GROUP_AGGREGATE,
        filters=[Filter(field=Field_.RESOLVED, op=Operator.EQ, value=True)],
        group_by=GroupField.AGENT_ID,
        metric=Metric(agg=Aggregation.AVG, field=Field_.CUSTOMER_RATING),
        sort=Sort(by="metric", direction=SortDirection.ASC),
        limit=1,
    )
    result = run(state, plan)
    # AGT-02 ratings 2 and 1 -> 1.5
    assert result.rows[0]["agent_id"] == "AGT-02"
    assert result.rows[0]["avg_customer_rating"] == 1.5


def test_having_min_count_excludes_small_groups(state: AppState) -> None:
    plan = QueryPlan(
        intent=Intent.GROUP_AGGREGATE,
        filters=[Filter(field=Field_.RESOLVED, op=Operator.EQ, value=True)],
        group_by=GroupField.AGENT_ID,
        metric=Metric(agg=Aggregation.AVG, field=Field_.CUSTOMER_RATING),
        having_min_count=3,
        sort=Sort(by="metric", direction=SortDirection.ASC),
    )
    result = run(state, plan)
    assert [row["agent_id"] for row in result.rows] == ["AGT-01"]


def test_ties_are_reported_rather_than_arbitrarily_broken(state: AppState) -> None:
    plan = QueryPlan(
        intent=Intent.GROUP_AGGREGATE,
        group_by=GroupField.CATEGORY,
        metric=Metric(agg=Aggregation.COUNT),
        sort=Sort(by="metric", direction=SortDirection.DESC),
        limit=1,
    )
    result = run(state, plan)
    # Billing 3, Technical 4, General 3 -> Technical wins outright
    assert result.rows[0]["category"] == "Technical"

    tie_plan = QueryPlan(
        intent=Intent.GROUP_AGGREGATE,
        filters=[
            Filter(field=Field_.CATEGORY, op=Operator.IN, value=["Billing", "General"])
        ],
        group_by=GroupField.CATEGORY,
        metric=Metric(agg=Aggregation.COUNT),
        sort=Sort(by="metric", direction=SortDirection.DESC),
        limit=1,
    )
    tie_result = run(state, tie_plan)
    assert len(tie_result.rows) == 2
    assert any("Tied at the top" in note for note in tie_result.notes)
    assert "tied" in format_answer(tie_plan, tie_result)


def test_category_with_highest_average_resolution_time(state: AppState) -> None:
    plan = QueryPlan(
        intent=Intent.GROUP_AGGREGATE,
        group_by=GroupField.CATEGORY,
        metric=Metric(agg=Aggregation.AVG, field=Field_.RESOLUTION_TIME_HRS),
        sort=Sort(by="metric", direction=SortDirection.DESC),
        limit=1,
    )
    result = run(state, plan)
    # Technical resolved: 4 and 90 -> 47.0
    assert result.rows[0]["category"] == "Technical"
    assert result.rows[0]["avg_resolution_time_hrs"] == 47.0


def test_groups_without_a_metric_value_rank_last(state: AppState) -> None:
    """An agent with no rated tickets is not "the lowest rated"."""
    plan = QueryPlan(
        intent=Intent.GROUP_AGGREGATE,
        group_by=GroupField.STATUS,
        metric=Metric(agg=Aggregation.AVG, field=Field_.CUSTOMER_RATING),
        sort=Sort(by="metric", direction=SortDirection.ASC),
    )
    result = run(state, plan)
    assert result.rows[0]["status"] == "Resolved"
    assert result.rows[-1]["avg_customer_rating"] is None


# -- comparison and relative outliers --------------------------------------


def test_compare_two_priorities(state: AppState) -> None:
    plan = QueryPlan(
        intent=Intent.COMPARE,
        metric=Metric(agg=Aggregation.AVG, field=Field_.RESOLUTION_TIME_HRS),
        compare=[
            CompareArm(
                label="High",
                filters=[Filter(field=Field_.PRIORITY, op=Operator.EQ, value="High")],
            ),
            CompareArm(
                label="Critical",
                filters=[
                    Filter(field=Field_.PRIORITY, op=Operator.EQ, value="Critical")
                ],
            ),
        ],
    )
    result = run(state, plan)
    by_label = {row["group"]: row for row in result.rows}
    assert by_label["High"]["avg_resolution_time_hrs"] == 3.0  # 2.0 and 4.0
    assert by_label["Critical"]["avg_resolution_time_hrs"] == 90.0
    assert "Critical is higher" in format_answer(plan, result)


def test_relative_outliers_requires_all_conditions(state: AppState) -> None:
    plan = QueryPlan(
        intent=Intent.RELATIVE_OUTLIERS,
        filters=[Filter(field=Field_.RESOLVED, op=Operator.EQ, value=True)],
        group_by=GroupField.AGENT_ID,
        relative_conditions=[
            RelativeCondition(
                metric=Metric(agg=Aggregation.AVG, field=Field_.CUSTOMER_RATING),
                direction=Direction.BELOW,
            ),
            RelativeCondition(
                metric=Metric(
                    agg=Aggregation.AVG, field=Field_.RESOLUTION_TIME_HRS
                ),
                direction=Direction.ABOVE,
            ),
        ],
    )
    result = run(state, plan)
    # Overall (resolved): avg rating 3.17, avg resolution 18.33.
    # AGT-01 4.0 / 3.67 -> no. AGT-02 1.5 / 46.5 -> yes. AGT-03 4.0 / 6.0 -> no.
    assert [row["agent_id"] for row in result.rows] == ["AGT-02"]
    assert result.overall["avg_customer_rating"] == 3.17


# -- guardrails -------------------------------------------------------------


def test_unsupported_intent_is_rejected_by_the_engine(state: AppState) -> None:
    plan = QueryPlan(intent=Intent.UNSUPPORTED, reason="out of scope")
    with pytest.raises(QueryPlanError):
        run(state, plan)


def test_plan_rejects_unknown_category_value() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        Filter(field=Field_.CATEGORY, op=Operator.EQ, value="Hardware")


def test_plan_rejects_non_numeric_average() -> None:
    with pytest.raises(ValueError, match="numeric field"):
        Metric(agg=Aggregation.AVG, field=Field_.CATEGORY)


def test_plan_rejects_sql_injection_in_a_filter_value() -> None:
    with pytest.raises(ValueError):
        Filter(field=Field_.STATUS, op=Operator.EQ, value="Open'; DROP TABLE tickets;--")


def test_injection_in_a_free_text_filter_is_parameterised(state: AppState) -> None:
    """Free-text fields accept any string, so it must be bound, never inlined."""
    plan = QueryPlan(
        intent=Intent.COUNT,
        filters=[
            Filter(
                field=Field_.ISSUE_SUMMARY,
                op=Operator.CONTAINS,
                value="x'; DROP TABLE tickets;--",
            )
        ],
    )
    assert run(state, plan).scalar == 0
    assert state.db.count_tickets() == 10


def test_group_aggregate_requires_group_by() -> None:
    with pytest.raises(ValueError, match="requires group_by"):
        QueryPlan(intent=Intent.GROUP_AGGREGATE)


def test_compare_requires_two_arms() -> None:
    with pytest.raises(ValueError, match="at least two compare arms"):
        QueryPlan(
            intent=Intent.COMPARE,
            compare=[CompareArm(label="High", filters=[])],
        )


def test_row_limit_is_capped_by_settings(state: AppState) -> None:
    state.settings.max_row_limit = 3
    plan = QueryPlan(intent=Intent.LIST, limit=200)
    result = run(state, plan)
    assert len(result.rows) == 3
    assert result.row_limit_applied == 3
