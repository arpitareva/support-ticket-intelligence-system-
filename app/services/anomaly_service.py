"""Deterministic anomaly detection.

No LLM is involved. Every rule states the metric it used, the threshold it
compared against, and how that threshold was derived, so a reviewer can
reproduce any flag by hand.

Reference time
--------------
The dataset is historical (Jan-Mar 2024). Measuring ticket age against
``datetime.now()`` would make every unresolved ticket years old and the
"older than 24 hours" rule vacuous. The default reference time is therefore
the latest ``created_at`` in the dataset - the moment the extract was taken.
``ANOMALY_REFERENCE_TIME`` can be set to ``now`` or to a fixed ISO timestamp
for a live deployment. Whichever is used is reported in the API response.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from app.config import Settings
from app.database.database import TicketDatabase
from app.schemas.response import Anomaly, AnomalyResponse, AnomalySummary

logger = logging.getLogger(__name__)

HIGH_PRIORITIES = ("High", "Critical")
SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


class AnomalyService:
    def __init__(self, db: TicketDatabase, settings: Settings) -> None:
        self.db = db
        self.settings = settings

    # -- reference time ----------------------------------------------------

    def resolve_reference_time(self, df: pd.DataFrame) -> tuple[datetime, str]:
        configured = self.settings.anomaly_reference_time
        if configured == "now":
            return datetime.now(), "system clock (ANOMALY_REFERENCE_TIME=now)"
        if configured != "max_created_at":
            try:
                return (
                    datetime.fromisoformat(configured),
                    f"fixed timestamp from configuration ({configured})",
                )
            except ValueError:
                logger.warning(
                    "Invalid ANOMALY_REFERENCE_TIME=%r; falling back to the "
                    "latest ticket timestamp",
                    configured,
                )
        return (
            df["created_at"].max().to_pydatetime(),
            "latest created_at in the dataset (dataset extract time)",
        )

    # -- public API --------------------------------------------------------

    def detect(self, types: list[str] | None = None) -> AnomalyResponse:
        df = self.db.load_dataframe()
        reference_time, basis = self.resolve_reference_time(df)
        resolved_times = df.loc[df["resolution_time_hrs"].notna(), "resolution_time_hrs"]

        thresholds: dict[str, Any] = {
            "reference_time_basis": basis,
            "aging_threshold_hours": self.settings.aging_threshold_hours,
        }

        anomalies: list[Anomaly] = []
        anomalies += self._aging_high_priority(df, reference_time, thresholds)
        anomalies += self._long_resolution(df, resolved_times, thresholds)
        anomalies += self._slow_response_high_priority(df, thresholds)
        anomalies += self._low_customer_rating(df, thresholds)
        anomalies += self._agent_outliers(df, thresholds)
        anomalies += self._data_quality(df, thresholds)

        if types:
            wanted = {t.strip() for t in types if t.strip()}
            anomalies = [a for a in anomalies if a.anomaly_type in wanted]

        anomalies.sort(
            key=lambda a: (SEVERITY_ORDER.get(a.severity, 9), -a.value)
        )

        by_type: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        for a in anomalies:
            by_type[a.anomaly_type] = by_type.get(a.anomaly_type, 0) + 1
            by_severity[a.severity] = by_severity.get(a.severity, 0) + 1

        return AnomalyResponse(
            summary=AnomalySummary(
                total=len(anomalies),
                by_type=by_type,
                by_severity=by_severity,
                reference_time=reference_time,
                thresholds=thresholds,
            ),
            anomalies=anomalies,
        )

    def count(self) -> int:
        return self.detect().summary.total

    # -- rules -------------------------------------------------------------

    def _aging_high_priority(
        self,
        df: pd.DataFrame,
        reference_time: datetime,
        thresholds: dict[str, Any],
    ) -> list[Anomaly]:
        """A. Unresolved High/Critical tickets older than the ageing threshold."""
        limit = self.settings.aging_threshold_hours
        subset = df[
            df["priority"].isin(HIGH_PRIORITIES) & (df["resolved"] == 0)
        ].copy()
        if subset.empty:
            return []
        age = (pd.Timestamp(reference_time) - subset["created_at"]).dt.total_seconds() / 3600
        subset = subset.assign(age_hours=age)
        breached = subset[subset["age_hours"] > limit]

        out: list[Anomaly] = []
        for row in breached.itertuples():
            age_hours = round(float(row.age_hours), 1)
            # Age buckets keep the rule informative on a historical extract,
            # where almost every open ticket is past 24h.
            if age_hours >= 7 * 24 and row.priority == "Critical":
                severity = "critical"
            elif age_hours >= 7 * 24 or row.priority == "Critical":
                severity = "high"
            elif age_hours >= 3 * 24:
                severity = "medium"
            else:
                severity = "low"
            out.append(
                Anomaly(
                    ticket_id=row.ticket_id,
                    entity_id=row.ticket_id,
                    anomaly_type="unresolved_high_priority_aging",
                    severity=severity,
                    metric="age_hours_since_creation",
                    value=age_hours,
                    threshold=float(limit),
                    threshold_basis=(
                        f"{limit:g}h SLA for High/Critical tickets, aged against "
                        f"{reference_time:%Y-%m-%d %H:%M}"
                    ),
                    reason=(
                        f"{row.priority} ticket still {row.status.lower()} after "
                        f"{age_hours}h (SLA {limit:g}h)."
                    ),
                    context={
                        "priority": row.priority,
                        "status": row.status,
                        "category": row.category,
                        "agent_id": row.agent_id,
                        "created_at": row.created_at.isoformat(sep=" "),
                        "response_time_hrs": _nan_to_none(row.response_time_hrs),
                    },
                )
            )
        return out

    def _long_resolution(
        self,
        df: pd.DataFrame,
        resolved_times: pd.Series,
        thresholds: dict[str, Any],
    ) -> list[Anomaly]:
        """B. Resolution times above the Nth percentile of resolved tickets."""
        if len(resolved_times) < 20:
            logger.info("Too few resolved tickets for a percentile threshold")
            return []
        pct = self.settings.long_resolution_percentile
        threshold = float(np.percentile(resolved_times, pct))
        extreme = float(np.percentile(resolved_times, 99))
        thresholds["long_resolution_hours"] = round(threshold, 2)
        thresholds["long_resolution_percentile"] = pct
        thresholds["long_resolution_sample_size"] = int(len(resolved_times))

        breached = df[df["resolution_time_hrs"] > threshold]
        out: list[Anomaly] = []
        for row in breached.itertuples():
            value = round(float(row.resolution_time_hrs), 2)
            severity = "critical" if value > extreme else "high"
            out.append(
                Anomaly(
                    ticket_id=row.ticket_id,
                    entity_id=row.ticket_id,
                    anomaly_type="long_resolution",
                    severity=severity,
                    metric="resolution_time_hrs",
                    value=value,
                    threshold=round(threshold, 2),
                    threshold_basis=(
                        f"p{pct:g} of the {len(resolved_times)} resolved tickets "
                        "(unresolved/null excluded)"
                    ),
                    reason=(
                        f"Resolution took {value}h, above the p{pct:g} threshold of "
                        f"{round(threshold, 2)}h."
                    ),
                    context={
                        "priority": row.priority,
                        "category": row.category,
                        "agent_id": row.agent_id,
                        "customer_rating": _nan_to_none(row.customer_rating),
                    },
                )
            )
        return out

    def _slow_response_high_priority(
        self, df: pd.DataFrame, thresholds: dict[str, Any]
    ) -> list[Anomaly]:
        """C1. High/Critical tickets whose first response was unusually slow."""
        response = df.loc[df["response_time_hrs"].notna(), "response_time_hrs"]
        if len(response) < 20:
            return []
        pct = self.settings.slow_response_percentile
        threshold = float(np.percentile(response, pct))
        thresholds["slow_response_hours"] = round(threshold, 2)
        thresholds["slow_response_percentile"] = pct

        breached = df[
            df["priority"].isin(HIGH_PRIORITIES)
            & (df["response_time_hrs"] > threshold)
        ]
        out: list[Anomaly] = []
        for row in breached.itertuples():
            value = round(float(row.response_time_hrs), 2)
            severity = "high" if row.priority == "Critical" else "medium"
            out.append(
                Anomaly(
                    ticket_id=row.ticket_id,
                    entity_id=row.ticket_id,
                    anomaly_type="slow_first_response_high_priority",
                    severity=severity,
                    metric="response_time_hrs",
                    value=value,
                    threshold=round(threshold, 2),
                    threshold_basis=f"p{pct:g} of response times across all tickets",
                    reason=(
                        f"{row.priority} ticket waited {value}h for a first "
                        f"response, above the p{pct:g} threshold of "
                        f"{round(threshold, 2)}h."
                    ),
                    context={
                        "priority": row.priority,
                        "status": row.status,
                        "agent_id": row.agent_id,
                    },
                )
            )
        return out

    def _low_customer_rating(
        self, df: pd.DataFrame, thresholds: dict[str, Any]
    ) -> list[Anomaly]:
        """C2. Resolved tickets with a very low satisfaction rating."""
        limit = self.settings.low_rating_threshold
        thresholds["low_rating_at_or_below"] = limit
        breached = df[df["customer_rating"].notna() & (df["customer_rating"] <= limit)]
        out: list[Anomaly] = []
        for row in breached.itertuples():
            value = float(row.customer_rating)
            severity = "high" if value <= 1 else "medium"
            out.append(
                Anomaly(
                    ticket_id=row.ticket_id,
                    entity_id=row.ticket_id,
                    anomaly_type="low_customer_rating",
                    severity=severity,
                    metric="customer_rating",
                    value=value,
                    threshold=float(limit),
                    threshold_basis=f"ratings at or below {limit} on a 1-5 scale",
                    reason=(
                        f"Customer rated this ticket {int(value)}/5 after "
                        f"{_fmt_hours(row.resolution_time_hrs)} to resolve."
                    ),
                    context={
                        "priority": row.priority,
                        "category": row.category,
                        "agent_id": row.agent_id,
                        "resolution_time_hrs": _nan_to_none(row.resolution_time_hrs),
                    },
                )
            )
        return out

    def _agent_outliers(
        self, df: pd.DataFrame, thresholds: dict[str, Any]
    ) -> list[Anomaly]:
        """C3. Agent-level outliers, one standard deviation from the mean.

        Agent-level rather than ticket-level, so ``ticket_id`` is null and
        ``entity_type`` is ``agent``. Agents below the minimum ticket count are
        excluded because small samples produce meaningless averages.
        """
        min_tickets = self.settings.min_tickets_for_agent_outlier
        resolved = df[df["resolved"] == 1]
        if resolved.empty:
            return []
        grouped = resolved.groupby("agent_id").agg(
            resolved_tickets=("ticket_id", "count"),
            avg_rating=("customer_rating", "mean"),
            avg_resolution=("resolution_time_hrs", "mean"),
        )
        eligible = grouped[grouped["resolved_tickets"] >= min_tickets]
        if len(eligible) < 3:
            return []

        rating_mean = float(eligible["avg_rating"].mean())
        rating_std = float(eligible["avg_rating"].std(ddof=0))
        resolution_mean = float(eligible["avg_resolution"].mean())
        resolution_std = float(eligible["avg_resolution"].std(ddof=0))
        thresholds["agent_outlier_min_resolved_tickets"] = min_tickets
        thresholds["agent_rating_floor"] = round(rating_mean - rating_std, 2)
        thresholds["agent_resolution_ceiling"] = round(
            resolution_mean + resolution_std, 2
        )

        out: list[Anomaly] = []
        for agent_id, row in eligible.iterrows():
            if rating_std > 0 and row["avg_rating"] < rating_mean - rating_std:
                out.append(
                    Anomaly(
                        ticket_id=None,
                        entity_type="agent",
                        entity_id=str(agent_id),
                        anomaly_type="agent_rating_outlier",
                        severity="medium",
                        metric="avg_customer_rating",
                        value=round(float(row["avg_rating"]), 2),
                        threshold=round(rating_mean - rating_std, 2),
                        threshold_basis=(
                            f"one standard deviation below the cross-agent mean "
                            f"({round(rating_mean, 2)} - {round(rating_std, 2)})"
                        ),
                        reason=(
                            f"{agent_id} averages "
                            f"{round(float(row['avg_rating']), 2)}/5 across "
                            f"{int(row['resolved_tickets'])} resolved tickets, "
                            "more than one standard deviation below peers."
                        ),
                        context={"resolved_tickets": int(row["resolved_tickets"])},
                    )
                )
            if resolution_std > 0 and row["avg_resolution"] > resolution_mean + resolution_std:
                out.append(
                    Anomaly(
                        ticket_id=None,
                        entity_type="agent",
                        entity_id=str(agent_id),
                        anomaly_type="agent_resolution_outlier",
                        severity="medium",
                        metric="avg_resolution_time_hrs",
                        value=round(float(row["avg_resolution"]), 2),
                        threshold=round(resolution_mean + resolution_std, 2),
                        threshold_basis=(
                            f"one standard deviation above the cross-agent mean "
                            f"({round(resolution_mean, 2)} + {round(resolution_std, 2)})"
                        ),
                        reason=(
                            f"{agent_id} averages "
                            f"{round(float(row['avg_resolution']), 2)}h to resolve "
                            f"across {int(row['resolved_tickets'])} tickets, more "
                            "than one standard deviation above peers."
                        ),
                        context={"resolved_tickets": int(row["resolved_tickets"])},
                    )
                )
        return out

    def _data_quality(
        self, df: pd.DataFrame, thresholds: dict[str, Any]
    ) -> list[Anomaly]:
        """C4. Records that contradict the documented column semantics.

        ``resolution_time_hrs`` is hours from creation to resolution and
        ``response_time_hrs`` hours from creation to first response, so
        resolution cannot precede the first response. Rows that do are flagged
        because they distort any resolution-time statistic.
        """
        breached = df[
            df["resolution_time_hrs"].notna()
            & (df["resolution_time_hrs"] < df["response_time_hrs"])
        ]
        if breached.empty:
            return []
        thresholds["data_quality_rule"] = (
            "resolution_time_hrs must be >= response_time_hrs"
        )
        out: list[Anomaly] = []
        for row in breached.itertuples():
            gap = round(
                float(row.response_time_hrs) - float(row.resolution_time_hrs), 2
            )
            out.append(
                Anomaly(
                    ticket_id=row.ticket_id,
                    entity_id=row.ticket_id,
                    anomaly_type="data_quality_inconsistent_timings",
                    severity="low",
                    metric="response_minus_resolution_hrs",
                    value=gap,
                    threshold=0.0,
                    threshold_basis="resolution_time_hrs >= response_time_hrs",
                    reason=(
                        f"Resolution recorded at {row.resolution_time_hrs}h, "
                        f"before the first response at {row.response_time_hrs}h."
                    ),
                    context={
                        "response_time_hrs": _nan_to_none(row.response_time_hrs),
                        "resolution_time_hrs": _nan_to_none(row.resolution_time_hrs),
                        "agent_id": row.agent_id,
                    },
                )
            )
        return out


def _nan_to_none(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return float(value)


def _fmt_hours(value: Any) -> str:
    cleaned = _nan_to_none(value)
    return "an unknown time" if cleaned is None else f"{round(cleaned, 1)}h"
