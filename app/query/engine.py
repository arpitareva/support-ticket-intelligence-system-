"""Deterministic query engine.

Every number the system reports is produced here, from SQL over SQLite (plus
NumPy for percentiles, which SQLite lacks). The LLM's only influence is the
validated ``QueryPlan``; this module maps plan enums to SQL fragments through
lookup tables, so no user-supplied string ever becomes SQL. All literals are
bound as parameters.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.config import Settings
from app.database.database import TicketDatabase
from app.database.models import TABLE_NAME
from app.schemas.query_plan import (
    Aggregation,
    CompareArm,
    DetailSpec,
    Direction,
    Field_,
    Filter,
    GroupField,
    Intent,
    Metric,
    Operator,
    QueryPlan,
    SortDirection,
    TimeRange,
)
from app.utils.errors import QueryPlanError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Whitelists: the only bridge between plan enums and SQL text.
# ---------------------------------------------------------------------------

_COLUMN_SQL: dict[Field_, str] = {
    Field_.TICKET_ID: "ticket_id",
    Field_.CREATED_AT: "created_at",
    Field_.CATEGORY: "category",
    Field_.PRIORITY: "priority",
    Field_.STATUS: "status",
    Field_.RESPONSE_TIME_HRS: "response_time_hrs",
    Field_.RESOLUTION_TIME_HRS: "resolution_time_hrs",
    Field_.AGENT_ID: "agent_id",
    Field_.CUSTOMER_RATING: "customer_rating",
    Field_.ISSUE_SUMMARY: "issue_summary",
    Field_.RESOLVED: "resolved",
}

_GROUP_SQL: dict[GroupField, str] = {
    GroupField.CATEGORY: "category",
    GroupField.PRIORITY: "priority",
    GroupField.STATUS: "status",
    GroupField.AGENT_ID: "agent_id",
    GroupField.DAY: "substr(created_at, 1, 10)",
    GroupField.WEEK: "strftime('%Y-W%W', created_at)",
    GroupField.MONTH: "substr(created_at, 1, 7)",
}

_SQL_AGGREGATIONS: dict[Aggregation, str] = {
    Aggregation.AVG: "AVG",
    Aggregation.SUM: "SUM",
    Aggregation.MIN: "MIN",
    Aggregation.MAX: "MAX",
}

_PERCENTILE_AGGREGATIONS = {Aggregation.MEDIAN: 50.0, Aggregation.P95: 95.0}

LIST_COLUMNS = [
    "ticket_id",
    "created_at",
    "category",
    "priority",
    "status",
    "response_time_hrs",
    "resolution_time_hrs",
    "agent_id",
    "customer_rating",
    "issue_summary",
]


@dataclass
class QueryResult:
    """Outcome of executing a plan. Consumed by the answer formatter."""

    result_type: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    # Stage-one group summary for two-stage (group_detail) results; ``rows``
    # then holds the detail rows belonging to those groups.
    group_rows: list[dict[str, Any]] = field(default_factory=list)
    scalar: float | int | None = None
    matched_rows: int = 0
    sample_size: int | None = None
    overall: dict[str, float | None] = field(default_factory=dict)
    sql: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    row_limit_applied: int | None = None
    latency_ms: int = 0


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 2)
    return value


class QueryEngine:
    def __init__(self, db: TicketDatabase, settings: Settings) -> None:
        self.db = db
        self.settings = settings
        self._last_group_sql: str = ""

    # -- WHERE clause construction -----------------------------------------

    def _compile_filter(self, f: Filter) -> tuple[str, list[Any]]:
        col = _COLUMN_SQL[f.field]
        value = f.value

        if f.field is Field_.RESOLVED:
            flag = 1 if value else 0
            if f.op is Operator.NE:
                flag = 1 - flag
            return f"{col} = ?", [flag]

        if f.op is Operator.IS_NULL:
            return f"{col} IS NULL", []
        if f.op is Operator.NOT_NULL:
            return f"{col} IS NOT NULL", []
        if f.op is Operator.CONTAINS:
            escaped = str(value).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            return f"{col} LIKE ? ESCAPE '\\'", [f"%{escaped}%"]
        if f.op is Operator.IN:
            marks = ", ".join(["?"] * len(value))
            return f"{col} IN ({marks})", [self._bind(f.field, v) for v in value]
        if f.op is Operator.NOT_IN:
            marks = ", ".join(["?"] * len(value))
            return f"{col} NOT IN ({marks})", [self._bind(f.field, v) for v in value]
        if f.op is Operator.BETWEEN:
            lo, hi = value
            return f"{col} BETWEEN ? AND ?", [
                self._bind(f.field, lo),
                self._bind(f.field, hi),
            ]

        operator_sql = {
            Operator.EQ: "=",
            Operator.NE: "!=",
            Operator.GT: ">",
            Operator.GTE: ">=",
            Operator.LT: "<",
            Operator.LTE: "<=",
        }[f.op]
        return f"{col} {operator_sql} ?", [self._bind(f.field, value)]

    @staticmethod
    def _bind(field_: Field_, value: Any) -> Any:
        """Normalise a literal for binding (datetimes -> stored text format)."""
        if field_ is Field_.CREATED_AT:
            return str(value).replace("T", " ")[:19]
        return value

    def _where(
        self,
        filters: list[Filter],
        time_range: TimeRange,
        any_of: list[Filter] | None = None,
    ) -> tuple[str, list[Any]]:
        """Build ``WHERE`` from ANDed filters plus one optional OR block.

        ``any_of`` exists because some real questions are genuinely
        disjunctive - "Critical tickets not resolved within 12 hours" covers
        both slow resolutions and tickets still open. It is a single OR group
        ANDed with the rest, which is enough for those cases without turning
        the plan into an arbitrary expression tree.
        """
        clauses: list[str] = []
        params: list[Any] = []
        for f in filters:
            clause, bound = self._compile_filter(f)
            clauses.append(clause)
            params.extend(bound)
        if any_of:
            or_clauses: list[str] = []
            for f in any_of:
                clause, bound = self._compile_filter(f)
                or_clauses.append(clause)
                params.extend(bound)
            clauses.append("(" + " OR ".join(or_clauses) + ")")
        if time_range.start is not None:
            clauses.append("created_at >= ?")
            params.append(time_range.start.strftime("%Y-%m-%d %H:%M:%S"))
        if time_range.end is not None:
            clauses.append("created_at <= ?")
            params.append(time_range.end.strftime("%Y-%m-%d %H:%M:%S"))
        return (" AND ".join(clauses) if clauses else "1=1"), params

    # -- metric helpers ----------------------------------------------------

    def _metric_select(self, metric: Metric, alias: str) -> str | None:
        """SQL projection for a metric, or None if it needs NumPy."""
        if metric.agg is Aggregation.COUNT:
            return f"COUNT(*) AS {alias}"
        if metric.agg is Aggregation.COUNT_DISTINCT:
            return f"COUNT(DISTINCT {_COLUMN_SQL[metric.field]}) AS {alias}"
        if metric.agg in _SQL_AGGREGATIONS:
            return f"{_SQL_AGGREGATIONS[metric.agg]}({_COLUMN_SQL[metric.field]}) AS {alias}"
        return None

    def _percentile(self, values: list[float], pct: float) -> float | None:
        clean = [v for v in values if v is not None and not np.isnan(v)]
        if not clean:
            return None
        return float(np.percentile(clean, pct))

    def _scalar_metric(
        self, metric: Metric, where: str, params: list[Any]
    ) -> tuple[float | None, int, str]:
        """Compute one metric over the filtered set. Returns (value, n, sql)."""
        if metric.agg in _PERCENTILE_AGGREGATIONS:
            col = _COLUMN_SQL[metric.field]
            sql = f"SELECT {col} AS v FROM {TABLE_NAME} WHERE {where} AND {col} IS NOT NULL"
            rows = self.db.fetch_all(sql, params)
            values = [float(r["v"]) for r in rows]
            return (
                self._percentile(values, _PERCENTILE_AGGREGATIONS[metric.agg]),
                len(values),
                sql,
            )

        projection = self._metric_select(metric, "value")
        count_col = (
            "COUNT(*)"
            if metric.agg is Aggregation.COUNT
            else f"COUNT({_COLUMN_SQL[metric.field]})"
        )
        sql = (
            f"SELECT {projection}, {count_col} AS n FROM {TABLE_NAME} WHERE {where}"
        )
        row = self.db.fetch_one(sql, params) or {}
        value = row.get("value")
        return (
            float(value) if value is not None else None,
            int(row.get("n") or 0),
            sql,
        )

    # -- public API --------------------------------------------------------

    def execute(self, plan: QueryPlan) -> QueryResult:
        started = time.perf_counter()
        handlers = {
            Intent.COUNT: self._run_count,
            Intent.AGGREGATE: self._run_aggregate,
            Intent.LIST: self._run_list,
            Intent.GROUP_AGGREGATE: self._run_group_aggregate,
            Intent.COMPARE: self._run_compare,
            Intent.RELATIVE_OUTLIERS: self._run_relative_outliers,
            Intent.GROUP_DETAIL: self._run_group_detail,
        }
        handler = handlers.get(plan.intent)
        if handler is None:
            raise QueryPlanError(
                "This question cannot be answered by the query engine",
                plan.reason or f"Unsupported intent '{plan.intent.value}'.",
            )
        result = handler(plan)
        result.latency_ms = int((time.perf_counter() - started) * 1000)
        return result

    def _matched(self, where: str, params: list[Any]) -> int:
        row = self.db.fetch_one(
            f"SELECT COUNT(*) AS n FROM {TABLE_NAME} WHERE {where}", params
        )
        return int(row["n"]) if row else 0

    def _run_count(self, plan: QueryPlan) -> QueryResult:
        where, params = self._where(plan.filters, plan.time_range, plan.any_of)
        sql = f"SELECT COUNT(*) AS value FROM {TABLE_NAME} WHERE {where}"
        row = self.db.fetch_one(sql, params) or {"value": 0}
        n = int(row["value"])
        return QueryResult(
            result_type="scalar", scalar=n, matched_rows=n, sample_size=n, sql=[sql]
        )

    def _run_aggregate(self, plan: QueryPlan) -> QueryResult:
        where, params = self._where(plan.filters, plan.time_range, plan.any_of)
        value, n, sql = self._scalar_metric(plan.metric, where, params)
        matched = self._matched(where, params)
        result = QueryResult(
            result_type="scalar",
            scalar=_round(value),
            matched_rows=matched,
            sample_size=n,
            sql=[sql],
        )
        if plan.metric.field is not None and n < matched:
            result.notes.append(
                f"{matched - n} of {matched} matching tickets have no "
                f"{plan.metric.field.value} value and were excluded."
            )
        if value is None:
            result.notes.append("No non-null values available for this metric.")
        return result

    def _run_list(self, plan: QueryPlan) -> QueryResult:
        where, params = self._where(plan.filters, plan.time_range, plan.any_of)
        limit = min(
            plan.limit or self.settings.default_row_limit, self.settings.max_row_limit
        )
        if plan.sort.by == "metric":
            order_col = "created_at"
        else:
            order_col = _COLUMN_SQL[Field_(plan.sort.by)]
        direction = "DESC" if plan.sort.direction is SortDirection.DESC else "ASC"
        sql = (
            f"SELECT {', '.join(LIST_COLUMNS)} FROM {TABLE_NAME} WHERE {where} "
            f"ORDER BY {order_col} {direction} LIMIT ?"
        )
        rows = self.db.fetch_all(sql, [*params, limit])
        matched = self._matched(where, params)
        result = QueryResult(
            result_type="table",
            rows=[{k: _round(v) for k, v in r.items()} for r in rows],
            matched_rows=matched,
            sample_size=matched,
            sql=[sql],
            row_limit_applied=limit,
        )
        if matched > len(rows):
            result.notes.append(
                f"Showing the first {len(rows)} of {matched} matching tickets."
            )
        return result

    def _rank_groups(
        self, plan: QueryPlan, where: str, params: list[Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, list[str]]:
        """Stage one of any grouped question: compute, filter and rank groups.

        Returns ``(selected, all_ranked, sort_key, notes)``. ``selected`` is the
        winning slice after ``limit``, widened to include every group tied at
        the top so a superlative never silently breaks a tie. Shared by
        ``group_aggregate`` and ``group_detail`` so both rank identically.
        """
        assert plan.group_by is not None
        group_sql = _GROUP_SQL[plan.group_by]
        metrics = [plan.metric, *plan.secondary_metrics]
        rows = self._grouped_metrics(
            group_sql, plan.group_by.value, metrics, where, params
        )
        notes: list[str] = []

        if plan.having_min_count:
            rows = [r for r in rows if r["ticket_count"] >= plan.having_min_count]
            notes.append(
                f"Only groups with at least {plan.having_min_count} matching "
                "tickets were considered."
            )

        sort_key = plan.metric.label if plan.sort.by == "metric" else plan.sort.by
        if sort_key not in (rows[0] if rows else {}):
            sort_key = plan.metric.label
        reverse = plan.sort.direction is SortDirection.DESC
        # Groups with no value for the metric (e.g. an agent with zero rated
        # tickets) are never "the lowest" - they are ranked last, not first.
        ranked = sorted(
            (r for r in rows if r.get(sort_key) is not None),
            key=lambda r: r[sort_key],
            reverse=reverse,
        )
        unranked = [r for r in rows if r.get(sort_key) is None]
        ordered = [*ranked, *unranked]

        limit = plan.limit or (min(len(ordered), self.settings.max_row_limit) or 1)
        selected = ordered[:limit]

        # Tie detection: honest reporting beats a confidently wrong "the" answer.
        if limit == 1 and len(ranked) > 1:
            top = ranked[0][sort_key]
            tied = [r for r in ranked if r[sort_key] == top]
            if len(tied) > 1:
                selected = tied
                notes.append(
                    "Tied at the top: "
                    + ", ".join(str(r[plan.group_by.value]) for r in tied)
                    + "."
                )
        if len(ordered) > len(selected):
            notes.append(f"{len(ordered)} groups matched in total.")
        return selected, ordered, sort_key, notes

    def _run_group_aggregate(self, plan: QueryPlan) -> QueryResult:
        where, params = self._where(plan.filters, plan.time_range, plan.any_of)
        selected, ordered, _, notes = self._rank_groups(plan, where, params)

        return QueryResult(
            result_type="group_table",
            rows=[{k: _round(v) for k, v in r.items()} for r in selected],
            matched_rows=self._matched(where, params),
            sql=[self._last_group_sql],
            notes=notes,
            row_limit_applied=len(selected) or None,
        )

    def _run_group_detail(self, plan: QueryPlan) -> QueryResult:
        """Two-stage question: an aggregate picks the group(s), then the
        individual rows belonging to them are retrieved.

        "Among unresolved High and Critical tickets, which agent has the most,
        and what are those ticket IDs?" is one query in intent but two in
        execution: rank agents, then fetch that agent's tickets. Expressing it
        as a single plan with an explicit second stage keeps the guarantee that
        no SQL text originates from the model - the group values used in the
        detail step come from stage one's own results, bound as parameters.
        """
        assert plan.group_by is not None
        where, params = self._where(plan.filters, plan.time_range, plan.any_of)
        selected, _, sort_key, notes = self._rank_groups(plan, where, params)
        statements = [self._last_group_sql]

        group_alias = plan.group_by.value
        group_sql = _GROUP_SQL[plan.group_by]
        winners = [r[group_alias] for r in selected]

        if not winners:
            return QueryResult(
                result_type="group_detail",
                group_rows=[],
                matched_rows=0,
                sql=statements,
                notes=notes,
            )

        detail = plan.detail or DetailSpec()
        detail_limit = min(detail.limit, self.settings.max_row_limit)
        order_col = (
            "created_at" if detail.sort.by == "metric" else _COLUMN_SQL[Field_(detail.sort.by)]
        )
        direction = "DESC" if detail.sort.direction is SortDirection.DESC else "ASC"
        marks = ", ".join(["?"] * len(winners))

        detail_where = f"{where} AND {group_sql} IN ({marks})"
        detail_params = [*params, *winners]
        detail_sql = (
            f"SELECT {', '.join(LIST_COLUMNS)} FROM {TABLE_NAME} "
            f"WHERE {detail_where} ORDER BY {order_col} {direction} LIMIT ?"
        )
        rows = self.db.fetch_all(detail_sql, [*detail_params, detail_limit])
        statements.append(detail_sql)
        total_detail = self._matched(detail_where, detail_params)

        if total_detail > len(rows):
            notes.append(
                f"Listing the first {len(rows)} of {total_detail} tickets for "
                f"{', '.join(str(w) for w in winners)}."
            )

        return QueryResult(
            result_type="group_detail",
            rows=[{k: _round(v) for k, v in r.items()} for r in rows],
            group_rows=[{k: _round(v) for k, v in r.items()} for r in selected],
            matched_rows=total_detail,
            sample_size=total_detail,
            sql=statements,
            notes=notes,
            row_limit_applied=detail_limit,
        )

    def _grouped_metrics(
        self,
        group_sql: str,
        group_alias: str,
        metrics: list[Metric],
        where: str,
        params: list[Any],
    ) -> list[dict[str, Any]]:
        """Compute one row per group with every requested metric."""
        projections = ["COUNT(*) AS ticket_count"]
        pending_percentiles: list[Metric] = []
        seen = {"ticket_count"}
        for m in metrics:
            alias = m.label
            if alias in seen:
                continue
            seen.add(alias)
            if m.agg in _PERCENTILE_AGGREGATIONS:
                pending_percentiles.append(m)
                continue
            projections.append(self._metric_select(m, alias))
            if m.field is not None:
                projections.append(f"COUNT({_COLUMN_SQL[m.field]}) AS {alias}__n")

        sql = (
            f"SELECT {group_sql} AS {group_alias}, {', '.join(projections)} "
            f"FROM {TABLE_NAME} WHERE {where} GROUP BY {group_sql}"
        )
        self._last_group_sql = sql
        rows = self.db.fetch_all(sql, params)

        for m in pending_percentiles:
            col = _COLUMN_SQL[m.field]
            pct = _PERCENTILE_AGGREGATIONS[m.agg]
            raw = self.db.fetch_all(
                f"SELECT {group_sql} AS {group_alias}, {col} AS v FROM {TABLE_NAME} "
                f"WHERE {where} AND {col} IS NOT NULL",
                params,
            )
            buckets: dict[Any, list[float]] = {}
            for r in raw:
                buckets.setdefault(r[group_alias], []).append(float(r["v"]))
            for r in rows:
                r[m.label] = self._percentile(buckets.get(r[group_alias], []), pct)

        return rows

    def _run_compare(self, plan: QueryPlan) -> QueryResult:
        rows: list[dict[str, Any]] = []
        statements: list[str] = []
        notes: list[str] = []
        for arm in plan.compare:
            where, params = self._where(
                [*plan.filters, *arm.filters], plan.time_range, plan.any_of
            )
            value, n, sql = self._scalar_metric(plan.metric, where, params)
            matched = self._matched(where, params)
            statements.append(sql)
            rows.append(
                {
                    "group": arm.label,
                    "ticket_count": matched,
                    plan.metric.label: _round(value),
                    "values_used": n,
                }
            )
        usable = [r for r in rows if r[plan.metric.label] is not None]
        if len(usable) >= 2:
            ordered = sorted(usable, key=lambda r: r[plan.metric.label], reverse=True)
            delta = ordered[0][plan.metric.label] - ordered[-1][plan.metric.label]
            notes.append(
                f"{ordered[0]['group']} is higher than {ordered[-1]['group']} "
                f"by {round(delta, 2)}."
            )
        return QueryResult(
            result_type="comparison",
            rows=rows,
            matched_rows=sum(r["ticket_count"] for r in rows),
            sql=statements,
            notes=notes,
        )

    def _run_relative_outliers(self, plan: QueryPlan) -> QueryResult:
        """Groups that sit above/below the overall value on every condition.

        Answers questions like "which agents have both a below-average rating
        and an above-average resolution time". The reference value for each
        condition is computed over the same filtered population, never per
        group, so the comparison is well defined.
        """
        assert plan.group_by is not None
        where, params = self._where(plan.filters, plan.time_range, plan.any_of)
        group_sql = _GROUP_SQL[plan.group_by]
        metrics = [c.metric for c in plan.relative_conditions]
        rows = self._grouped_metrics(
            group_sql, plan.group_by.value, metrics, where, params
        )
        statements = [self._last_group_sql]

        overall: dict[str, float | None] = {}
        for m in metrics:
            value, _, sql = self._scalar_metric(m, where, params)
            overall[m.label] = _round(value)
            statements.append(sql)

        if plan.having_min_count:
            rows = [r for r in rows if r["ticket_count"] >= plan.having_min_count]

        matching: list[dict[str, Any]] = []
        for r in rows:
            ok = True
            for cond in plan.relative_conditions:
                label = cond.metric.label
                ref = overall.get(label)
                val = r.get(label)
                if val is None or ref is None:
                    ok = False
                    break
                if cond.direction is Direction.ABOVE and not val > ref:
                    ok = False
                    break
                if cond.direction is Direction.BELOW and not val < ref:
                    ok = False
                    break
            if ok:
                matching.append({k: _round(v) for k, v in r.items()})

        sort_label = plan.relative_conditions[0].metric.label
        matching.sort(key=lambda r: r.get(sort_label) or 0)
        limit = min(plan.limit or self.settings.max_row_limit, self.settings.max_row_limit)
        notes = [
            "Reference values (overall, same filters): "
            + ", ".join(f"{k}={v}" for k, v in overall.items())
        ]
        if plan.having_min_count:
            notes.append(
                f"Only groups with at least {plan.having_min_count} matching "
                "tickets were considered."
            )
        return QueryResult(
            result_type="group_table",
            rows=matching[:limit],
            matched_rows=self._matched(where, params),
            overall=overall,
            sql=statements,
            notes=notes,
            row_limit_applied=limit,
        )

    # -- convenience readers used by /stats and the dashboard ---------------

    def dataset_stats(self) -> dict[str, Any]:
        def group_counts(column: str) -> dict[str, int]:
            rows = self.db.fetch_all(
                f"SELECT {column} AS k, COUNT(*) AS n FROM {TABLE_NAME} "
                f"GROUP BY {column} ORDER BY n DESC"
            )
            return {r["k"]: int(r["n"]) for r in rows}

        totals = self.db.fetch_one(
            f"""SELECT COUNT(*) AS total,
                       SUM(CASE WHEN resolved = 0 THEN 1 ELSE 0 END) AS unresolved,
                       SUM(CASE WHEN resolved = 1 THEN 1 ELSE 0 END) AS resolved,
                       SUM(CASE WHEN priority IN ('High','Critical') THEN 1 ELSE 0 END)
                           AS high_or_critical,
                       AVG(customer_rating) AS avg_rating,
                       COUNT(customer_rating) AS rated,
                       AVG(response_time_hrs) AS avg_response,
                       AVG(resolution_time_hrs) AS avg_resolution,
                       COUNT(DISTINCT agent_id) AS agents
                FROM {TABLE_NAME}"""
        ) or {}
        median_rows = self.db.fetch_all(
            f"SELECT resolution_time_hrs AS v FROM {TABLE_NAME} "
            "WHERE resolution_time_hrs IS NOT NULL"
        )
        median = self._percentile([float(r["v"]) for r in median_rows], 50.0)
        lo, hi = self.db.date_range()
        return {
            "total_tickets": int(totals.get("total") or 0),
            "by_status": group_counts("status"),
            "by_category": group_counts("category"),
            "by_priority": group_counts("priority"),
            "unresolved_tickets": int(totals.get("unresolved") or 0),
            "resolved_tickets": int(totals.get("resolved") or 0),
            "high_or_critical_tickets": int(totals.get("high_or_critical") or 0),
            "avg_customer_rating": _round(totals.get("avg_rating")),
            "avg_response_time_hrs": _round(totals.get("avg_response")),
            "avg_resolution_time_hrs": _round(totals.get("avg_resolution")),
            "median_resolution_time_hrs": _round(median),
            "rated_tickets": int(totals.get("rated") or 0),
            "agents": int(totals.get("agents") or 0),
            "date_range": {"start": lo, "end": hi},
        }

    def list_tickets(
        self,
        *,
        limit: int,
        offset: int,
        filters: list[Filter] | None = None,
    ) -> tuple[int, list[dict[str, Any]]]:
        where, params = self._where(filters or [], TimeRange())
        total = self._matched(where, params)
        rows = self.db.fetch_all(
            f"SELECT {', '.join(LIST_COLUMNS)} FROM {TABLE_NAME} WHERE {where} "
            "ORDER BY created_at DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        )
        return total, rows
