
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Intent(str, Enum):
    COUNT = "count"  # how many tickets match X
    LIST = "list"  # show me the tickets matching X
    AGGREGATE = "aggregate"  # one number over the whole filtered set
    GROUP_AGGREGATE = "group_aggregate"  # metric per category/agent/priority/...
    GROUP_DETAIL = "group_detail"  # rank groups, then list the winner's rows
    COMPARE = "compare"  # metric for two or more named subsets
    RELATIVE_OUTLIERS = "relative_outliers"  # groups above/below the overall mean
    UNSUPPORTED = "unsupported"  # out of scope; engine will not run


class Field_(str, Enum):
    """Filterable columns. ``resolved`` is a virtual boolean field."""

    TICKET_ID = "ticket_id"
    CREATED_AT = "created_at"
    CATEGORY = "category"
    PRIORITY = "priority"
    STATUS = "status"
    RESPONSE_TIME_HRS = "response_time_hrs"
    RESOLUTION_TIME_HRS = "resolution_time_hrs"
    AGENT_ID = "agent_id"
    CUSTOMER_RATING = "customer_rating"
    ISSUE_SUMMARY = "issue_summary"
    RESOLVED = "resolved"  # virtual: status == 'Resolved'


class Operator(str, Enum):
    EQ = "eq"
    NE = "ne"
    IN = "in"
    NOT_IN = "not_in"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    BETWEEN = "between"
    IS_NULL = "is_null"
    NOT_NULL = "not_null"
    CONTAINS = "contains"


class Aggregation(str, Enum):
    COUNT = "count"
    COUNT_DISTINCT = "count_distinct"
    AVG = "avg"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    P95 = "p95"


class GroupField(str, Enum):
    CATEGORY = "category"
    PRIORITY = "priority"
    STATUS = "status"
    AGENT_ID = "agent_id"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class SortDirection(str, Enum):
    ASC = "asc"
    DESC = "desc"


class Direction(str, Enum):
    ABOVE = "above"
    BELOW = "below"


NUMERIC_FIELDS = {
    Field_.RESPONSE_TIME_HRS,
    Field_.RESOLUTION_TIME_HRS,
    Field_.CUSTOMER_RATING,
}
TEXT_FIELDS = {
    Field_.TICKET_ID,
    Field_.CATEGORY,
    Field_.PRIORITY,
    Field_.STATUS,
    Field_.AGENT_ID,
    Field_.ISSUE_SUMMARY,
}

CATEGORIES = ["Billing", "Technical", "General"]
PRIORITIES = ["Low", "Medium", "High", "Critical"]
STATUSES = ["Open", "Resolved", "Escalated"]

_ALLOWED_VALUES: dict[Field_, list[str]] = {
    Field_.CATEGORY: CATEGORIES,
    Field_.PRIORITY: PRIORITIES,
    Field_.STATUS: STATUSES,
}


class Filter(BaseModel):
    """A single row-level predicate. Compiles to one parameterized clause."""

    model_config = ConfigDict(use_enum_values=False, extra="forbid")

    field: Field_
    op: Operator
    value: Any = None

    @model_validator(mode="after")
    def _check(self) -> "Filter":
        if self.op in (Operator.IS_NULL, Operator.NOT_NULL):
            self.value = None
            return self

        if self.value is None:
            raise ValueError(f"operator '{self.op.value}' requires a value")

        if self.op in (Operator.IN, Operator.NOT_IN):
            if not isinstance(self.value, list) or not self.value:
                raise ValueError("'in'/'not_in' require a non-empty list")
        elif self.op is Operator.BETWEEN:
            if not isinstance(self.value, list) or len(self.value) != 2:
                raise ValueError("'between' requires exactly two values")
        elif isinstance(self.value, list):
            raise ValueError(f"operator '{self.op.value}' takes a scalar value")

        if self.field is Field_.RESOLVED:
            if self.op not in (Operator.EQ, Operator.NE):
                raise ValueError("'resolved' supports only eq/ne")
            if not isinstance(self.value, bool):
                raise ValueError("'resolved' takes a boolean value")

        if self.op is Operator.CONTAINS and self.field not in TEXT_FIELDS:
            raise ValueError("'contains' applies to text fields only")

        if self.field in NUMERIC_FIELDS:
            for v in self.value if isinstance(self.value, list) else [self.value]:
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise ValueError(f"{self.field.value} expects numeric values")

        # Normalise/validate closed vocabularies case-insensitively so the LLM
        # saying "billing" does not silently return zero rows.
        allowed = _ALLOWED_VALUES.get(self.field)
        if allowed and self.op in (
            Operator.EQ,
            Operator.NE,
            Operator.IN,
            Operator.NOT_IN,
        ):
            lookup = {a.lower(): a for a in allowed}
            raw = self.value if isinstance(self.value, list) else [self.value]
            fixed = []
            for v in raw:
                if not isinstance(v, str) or v.lower() not in lookup:
                    raise ValueError(
                        f"{self.field.value} must be one of {allowed}, got {v!r}"
                    )
                fixed.append(lookup[v.lower()])
            self.value = fixed if isinstance(self.value, list) else fixed[0]
        return self


class Metric(BaseModel):
    """What to compute. ``count`` ignores ``field``."""

    model_config = ConfigDict(extra="forbid")

    agg: Aggregation = Aggregation.COUNT
    field: Field_ | None = None

    @model_validator(mode="after")
    def _check(self) -> "Metric":
        if self.agg is Aggregation.COUNT:
            self.field = None
            return self
        if self.field is None:
            raise ValueError(f"aggregation '{self.agg.value}' requires a field")
        if self.agg is Aggregation.COUNT_DISTINCT:
            return self
        if self.field not in NUMERIC_FIELDS:
            raise ValueError(
                f"aggregation '{self.agg.value}' requires a numeric field, "
                f"got '{self.field.value}'"
            )
        return self

    @property
    def label(self) -> str:
        if self.agg is Aggregation.COUNT:
            return "ticket_count"
        assert self.field is not None
        return f"{self.agg.value}_{self.field.value}"


class Sort(BaseModel):
    model_config = ConfigDict(extra="forbid")

    by: str = "metric"  # "metric" or a Field_ value
    direction: SortDirection = SortDirection.DESC

    @field_validator("by")
    @classmethod
    def _known(cls, v: str) -> str:
        if v != "metric" and v not in {f.value for f in Field_}:
            raise ValueError(f"cannot sort by '{v}'")
        return v


class TimeRange(BaseModel):
    """Inclusive-start, exclusive-end window on ``created_at``."""

    model_config = ConfigDict(extra="forbid")

    start: datetime | None = None
    end: datetime | None = None

    @model_validator(mode="after")
    def _ordered(self) -> "TimeRange":
        if self.start and self.end and self.start > self.end:
            raise ValueError("time_range.start must not be after time_range.end")
        return self

    def is_empty(self) -> bool:
        return self.start is None and self.end is None


class CompareArm(BaseModel):
    """One side of a comparison, e.g. {label: "High", filters: [...]}"""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=60)
    filters: list[Filter] = Field(default_factory=list, max_length=10)


class RelativeCondition(BaseModel):
    """"below-average rating", "above-average resolution time", ..."""

    model_config = ConfigDict(extra="forbid")

    metric: Metric
    direction: Direction


class DetailSpec(BaseModel):
    """Stage two of a ``group_detail`` plan: how to list the winner's rows.

    Separate from the plan's own ``sort``/``limit``, which govern how *groups*
    are ranked in stage one. Without the split, "which agent has the most
    tickets" (limit 1 group) and "list their tickets" (limit 50 rows) would
    fight over the same two fields.
    """

    model_config = ConfigDict(extra="forbid")

    sort: Sort = Field(default_factory=lambda: Sort(by="created_at"))
    limit: int = Field(default=50, ge=1, le=200)


class QueryPlan(BaseModel):
    """Validated, executable description of a user's question."""

    model_config = ConfigDict(extra="forbid")

    intent: Intent
    filters: list[Filter] = Field(default_factory=list, max_length=10)
    any_of: list[Filter] = Field(default_factory=list, max_length=6)
    group_by: GroupField | None = None
    metric: Metric = Field(default_factory=Metric)
    secondary_metrics: list[Metric] = Field(default_factory=list, max_length=4)
    having_min_count: int | None = Field(default=None, ge=1, le=500)
    sort: Sort = Field(default_factory=Sort)
    limit: int | None = Field(default=None, ge=1, le=200)
    time_range: TimeRange = Field(default_factory=TimeRange)
    compare: list[CompareArm] = Field(default_factory=list, max_length=6)
    relative_conditions: list[RelativeCondition] = Field(
        default_factory=list, max_length=4
    )
    detail: DetailSpec | None = None
    reason: str | None = Field(default=None, max_length=400)

    @model_validator(mode="after")
    def _coherent(self) -> "QueryPlan":
        if self.intent in (
            Intent.GROUP_AGGREGATE,
            Intent.RELATIVE_OUTLIERS,
            Intent.GROUP_DETAIL,
        ):
            if self.group_by is None:
                raise ValueError(f"intent '{self.intent.value}' requires group_by")
        if self.intent is Intent.COMPARE and len(self.compare) < 2:
            raise ValueError("intent 'compare' requires at least two compare arms")
        if self.intent is Intent.RELATIVE_OUTLIERS and not self.relative_conditions:
            raise ValueError(
                "intent 'relative_outliers' requires at least one relative_condition"
            )
        if len(self.any_of) == 1:
            # A one-element OR is just an AND term; normalise so the SQL and
            # the explanation stay simple.
            self.filters = [*self.filters, self.any_of[0]]
            self.any_of = []
        if self.intent is Intent.GROUP_DETAIL:
            # A drill-down is meaningless without a winner, so default to the
            # single top group rather than every group's rows at once.
            if self.limit is None:
                self.limit = 1
            if self.detail is None:
                self.detail = DetailSpec()
        if self.intent is Intent.COUNT:
            self.metric = Metric(agg=Aggregation.COUNT)
        if self.intent is Intent.UNSUPPORTED and not self.reason:
            self.reason = "The question is outside the scope of this dataset."
        return self


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=500)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be blank")
        return v
