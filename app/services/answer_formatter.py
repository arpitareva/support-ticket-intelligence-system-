"""Response generation.

The final sentence the user reads is assembled here from the *engine's*
numbers using templates. The LLM is deliberately not asked to write it: a
model that paraphrases "3.48" as "roughly 3.5, which is quite low" has
introduced an unverifiable claim. Templates cost some fluency and buy exact
numbers every time.
"""

from __future__ import annotations

from app.query.engine import QueryResult
from app.schemas.query_plan import (
    Aggregation,
    Direction,
    Field_,
    Filter,
    Intent,
    Metric,
    Operator,
    QueryPlan,
)

_METRIC_PHRASES: dict[Aggregation, str] = {
    Aggregation.COUNT: "ticket count",
    Aggregation.COUNT_DISTINCT: "distinct value count",
    Aggregation.AVG: "average",
    Aggregation.SUM: "total",
    Aggregation.MIN: "minimum",
    Aggregation.MAX: "maximum",
    Aggregation.MEDIAN: "median",
    Aggregation.P95: "95th percentile",
}

_FIELD_PHRASES: dict[Field_, str] = {
    Field_.RESPONSE_TIME_HRS: "response time",
    Field_.RESOLUTION_TIME_HRS: "resolution time",
    Field_.CUSTOMER_RATING: "customer rating",
    Field_.CREATED_AT: "creation time",
    Field_.CATEGORY: "category",
    Field_.PRIORITY: "priority",
    Field_.STATUS: "status",
    Field_.AGENT_ID: "agent",
    Field_.TICKET_ID: "ticket",
    Field_.ISSUE_SUMMARY: "issue summary",
    Field_.RESOLVED: "resolution state",
}

_OP_PHRASES: dict[Operator, str] = {
    Operator.EQ: "is",
    Operator.NE: "is not",
    Operator.IN: "is one of",
    Operator.NOT_IN: "is not one of",
    Operator.GT: "is greater than",
    Operator.GTE: "is at least",
    Operator.LT: "is less than",
    Operator.LTE: "is at most",
    Operator.BETWEEN: "is between",
    Operator.IS_NULL: "is missing",
    Operator.NOT_NULL: "is present",
    Operator.CONTAINS: "contains",
}

_UNITS: dict[Field_, str] = {
    Field_.RESPONSE_TIME_HRS: " hours",
    Field_.RESOLUTION_TIME_HRS: " hours",
}


def _metric_phrase(metric: Metric) -> str:
    if metric.agg is Aggregation.COUNT:
        return "number of tickets"
    field_name = _FIELD_PHRASES.get(metric.field, metric.field.value)
    if metric.agg is Aggregation.COUNT_DISTINCT:
        return f"number of distinct {field_name} values"
    return f"{_METRIC_PHRASES[metric.agg]} {field_name}"


def _unit(metric: Metric) -> str:
    if metric.agg in (Aggregation.COUNT, Aggregation.COUNT_DISTINCT):
        return ""
    return _UNITS.get(metric.field, "")


def describe_filter(f: Filter) -> str:
    field_name = _FIELD_PHRASES.get(f.field, f.field.value)
    if f.field is Field_.RESOLVED:
        resolved = bool(f.value) if f.op is Operator.EQ else not bool(f.value)
        return "resolved" if resolved else "not resolved"
    if f.op in (Operator.IS_NULL, Operator.NOT_NULL):
        return f"{field_name} {_OP_PHRASES[f.op]}"
    if f.op in (Operator.IN, Operator.NOT_IN):
        return f"{field_name} {_OP_PHRASES[f.op]} {', '.join(map(str, f.value))}"
    if f.op is Operator.BETWEEN:
        return f"{field_name} between {f.value[0]} and {f.value[1]}"
    return f"{field_name} {_OP_PHRASES[f.op]} {f.value}"


def describe_plan(plan: QueryPlan) -> str:
    """Human-readable rendering of the plan, shown in the API and the UI."""
    parts: list[str] = [f"intent: {plan.intent.value}"]
    if plan.intent is Intent.UNSUPPORTED:
        return f"intent: unsupported - {plan.reason}"
    if plan.intent is not Intent.COUNT:
        parts.append(f"metric: {_metric_phrase(plan.metric)}")
    conditions = [describe_filter(f) for f in plan.filters]
    if plan.any_of:
        conditions.append(
            "(" + " or ".join(describe_filter(f) for f in plan.any_of) + ")"
        )
    if plan.time_range.start or plan.time_range.end:
        lo = plan.time_range.start.isoformat(sep=" ") if plan.time_range.start else "start"
        hi = plan.time_range.end.isoformat(sep=" ") if plan.time_range.end else "end"
        conditions.append(f"created between {lo} and {hi}")
    if conditions:
        parts.append("where " + " and ".join(conditions))
    if plan.group_by:
        parts.append(f"grouped by {plan.group_by.value}")
    if plan.having_min_count:
        parts.append(f"groups with >= {plan.having_min_count} tickets")
    if plan.compare:
        parts.append("comparing " + " vs ".join(a.label for a in plan.compare))
    for cond in plan.relative_conditions:
        parts.append(
            f"{_metric_phrase(cond.metric)} {cond.direction.value} the overall value"
        )
    if (
        plan.intent in (Intent.LIST, Intent.GROUP_AGGREGATE, Intent.GROUP_DETAIL)
        and plan.limit
    ):
        parts.append(f"{plan.sort.direction.value} by {plan.sort.by}, top {plan.limit}")
    if plan.intent is Intent.GROUP_DETAIL and plan.detail:
        parts.append(
            f"then list matching tickets for the winning {plan.group_by.value} "
            f"({plan.detail.sort.direction.value} by {plan.detail.sort.by}, "
            f"max {plan.detail.limit})"
        )
    return "; ".join(parts)


def _filter_clause(plan: QueryPlan) -> str:
    bits = [describe_filter(f) for f in plan.filters]
    if plan.any_of:
        bits.append("(" + " or ".join(describe_filter(f) for f in plan.any_of) + ")")
    if plan.time_range.start or plan.time_range.end:
        bits.append("within the requested date range")
    return f" where {' and '.join(bits)}" if bits else ""


def format_answer(plan: QueryPlan, result: QueryResult) -> str:
    """Turn exact engine output into one or two plain sentences."""
    scope = _filter_clause(plan)

    if plan.intent is Intent.COUNT:
        n = int(result.scalar or 0)
        noun = "ticket" if n == 1 else "tickets"
        verb = "is" if n == 1 else "are"
        tail = scope if scope else " in the dataset"
        return f"There {verb} {n} {noun}{tail}."

    if plan.intent is Intent.AGGREGATE:
        if result.scalar is None:
            return f"No data is available to compute the {_metric_phrase(plan.metric)}{scope}."
        return (
            f"The {_metric_phrase(plan.metric)}{scope} is "
            f"{result.scalar}{_unit(plan.metric)} "
            f"(based on {result.sample_size} ticket"
            f"{'s' if result.sample_size != 1 else ''})."
        )

    if plan.intent is Intent.LIST:
        if not result.rows:
            return f"No tickets{scope}."
        ids = [str(r.get("ticket_id")) for r in result.rows[:5]]
        preview = ", ".join(ids)
        more = "" if result.matched_rows <= len(ids) else f", and {result.matched_rows - len(ids)} more"
        return (
            f"{result.matched_rows} ticket"
            f"{'s' if result.matched_rows != 1 else ''}{scope}: {preview}{more}."
        )

    if plan.intent is Intent.GROUP_AGGREGATE:
        if not result.rows:
            return f"No groups matched{scope}."
        label = plan.group_by.value
        metric_label = plan.metric.label
        top = result.rows[0]
        value = top.get(metric_label)
        unit = _unit(plan.metric)
        if plan.limit == 1 and len(result.rows) == 1:
            return (
                f"{top.get(label)} has the {'highest' if plan.sort.direction.value == 'desc' else 'lowest'} "
                f"{_metric_phrase(plan.metric)}{scope} at {value}{unit} "
                f"({top.get('ticket_count')} tickets)."
            )
        if plan.limit == 1:
            names = ", ".join(str(r.get(label)) for r in result.rows)
            return (
                f"{names} are tied for the "
                f"{'highest' if plan.sort.direction.value == 'desc' else 'lowest'} "
                f"{_metric_phrase(plan.metric)}{scope} at {value}{unit}."
            )
        lead = ", ".join(
            f"{r.get(label)} ({r.get(metric_label)}{unit})" for r in result.rows[:3]
        )
        return (
            f"{_metric_phrase(plan.metric).capitalize()} by {label}{scope}, "
            f"{'highest' if plan.sort.direction.value == 'desc' else 'lowest'} first: {lead}."
        )

    if plan.intent is Intent.GROUP_DETAIL:
        label = plan.group_by.value
        if not result.group_rows:
            return f"No {label} values matched{scope}."
        metric_label = plan.metric.label
        unit = _unit(plan.metric)
        superlative = "highest" if plan.sort.direction.value == "desc" else "lowest"
        names = ", ".join(str(r.get(label)) for r in result.group_rows)
        value = result.group_rows[0].get(metric_label)

        if len(result.group_rows) == 1:
            lead = (
                f"{names} has the {superlative} {_metric_phrase(plan.metric)}"
                f"{scope} at {value}{unit}."
            )
        else:
            lead = (
                f"{names} are tied for the {superlative} "
                f"{_metric_phrase(plan.metric)}{scope} at {value}{unit}."
            )

        if not result.rows:
            return f"{lead} No individual tickets to list."
        ids = [str(r.get("ticket_id")) for r in result.rows]
        shown = ", ".join(ids[:20])
        more = "" if len(ids) <= 20 else f", and {len(ids) - 20} more"
        noun = "ticket" if len(ids) == 1 else "tickets"
        return f"{lead} The {len(ids)} {noun}: {shown}{more}."

    if plan.intent is Intent.COMPARE:
        label = plan.metric.label
        usable = [r for r in result.rows if r.get(label) is not None]
        if not usable:
            return f"No data is available for the requested comparison{scope}."
        unit = _unit(plan.metric)
        parts = [
            f"{r['group']}: {r[label]}{unit} ({r['ticket_count']} tickets)"
            for r in result.rows
        ]
        sentence = f"{_metric_phrase(plan.metric).capitalize()}{scope} - " + "; ".join(parts) + "."
        if len(usable) >= 2:
            ordered = sorted(usable, key=lambda r: r[label], reverse=True)
            diff = round(ordered[0][label] - ordered[-1][label], 2)
            sentence += (
                f" {ordered[0]['group']} is higher by {diff}{unit}"
                f" than {ordered[-1]['group']}."
            )
        return sentence

    if plan.intent is Intent.RELATIVE_OUTLIERS:
        label = plan.group_by.value
        conditions = " and ".join(
            f"{_metric_phrase(c.metric)} {c.direction.value} the overall "
            f"{result.overall.get(c.metric.label)}"
            for c in plan.relative_conditions
        )
        if not result.rows:
            return f"No {label} values have {conditions}{scope}."
        names = ", ".join(str(r.get(label)) for r in result.rows)
        return (
            f"{len(result.rows)} {label} value"
            f"{'s' if len(result.rows) != 1 else ''} have {conditions}{scope}: {names}."
        )

    return "The query engine returned no interpretable result."
