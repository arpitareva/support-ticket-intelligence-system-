"""CSV ingestion and validation.

Pandas is used exactly once, at load time: read the file, coerce dtypes,
validate the schema, and report data-quality findings. After this step all
analytics run in SQL/Python against SQLite, so the data layer has a single
well-defined entry point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from app.schemas.query_plan import CATEGORIES, PRIORITIES, STATUSES
from app.utils.errors import DataLoadError

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS: list[str] = [
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

NUMERIC_COLUMNS = ["response_time_hrs", "resolution_time_hrs", "customer_rating"]
NON_NULL_COLUMNS = [
    "ticket_id",
    "created_at",
    "category",
    "priority",
    "status",
    "agent_id",
]
CATEGORICAL_DOMAINS = {
    "category": set(CATEGORIES),
    "priority": set(PRIORITIES),
    "status": set(STATUSES),
}


@dataclass
class LoadReport:
    """What the loader saw. Surfaced in logs and in the test suite."""

    rows: int = 0
    dropped_rows: int = 0
    null_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    min_created_at: pd.Timestamp | None = None
    max_created_at: pd.Timestamp | None = None


def load_tickets(csv_path: str | Path) -> tuple[pd.DataFrame, LoadReport]:
    """Read and validate the ticket CSV.

    Raises ``DataLoadError`` for problems that make the dataset unusable
    (missing file, missing columns, empty file, all dates invalid) and records
    recoverable issues as warnings on the returned report.
    """
    path = Path(csv_path)
    if not path.exists():
        raise DataLoadError(
            f"Ticket CSV not found at {path}",
            "Place support_tickets.csv in the data/ directory or set CSV_PATH.",
        )

    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=True)
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError(f"Ticket CSV at {path} is empty", str(exc)) from exc
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise DataLoadError(f"Could not parse ticket CSV at {path}", str(exc)) from exc

    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise DataLoadError(
            "Ticket CSV schema is invalid",
            f"Missing required column(s): {', '.join(missing)}",
        )

    report = LoadReport()
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if extra:
        report.warnings.append(f"Ignoring unexpected column(s): {', '.join(extra)}")
    df = df[EXPECTED_COLUMNS].copy()

    if df.empty:
        raise DataLoadError(
            "Ticket CSV contains no data rows",
            "The file has headers but zero records.",
        )

    for col in ["ticket_id", "category", "priority", "status", "agent_id"]:
        df[col] = df[col].str.strip()
    df["issue_summary"] = df["issue_summary"].fillna("").str.strip()

    # created_at -> datetime; rows with unparseable dates cannot participate in
    # time filtering or ageing, so they are dropped rather than silently kept.
    parsed = pd.to_datetime(df["created_at"], errors="coerce", format="mixed")
    bad_dates = int(parsed.isna().sum())
    if bad_dates:
        report.warnings.append(f"Dropped {bad_dates} row(s) with unparseable created_at")
        report.dropped_rows += bad_dates
    df["created_at"] = parsed
    df = df[df["created_at"].notna()].copy()
    if df.empty:
        raise DataLoadError(
            "No ticket rows have a valid created_at value",
            "Expected timestamps like '2024-01-03 09:12'.",
        )

    for col in NUMERIC_COLUMNS:
        coerced = pd.to_numeric(df[col], errors="coerce")
        newly_null = int((coerced.isna() & df[col].notna()).sum())
        if newly_null:
            report.warnings.append(
                f"{newly_null} non-numeric value(s) in {col} coerced to null"
            )
        df[col] = coerced

    missing_required = df[NON_NULL_COLUMNS].isna().any(axis=1)
    if int(missing_required.sum()):
        report.warnings.append(
            f"Dropped {int(missing_required.sum())} row(s) missing required identifiers"
        )
        report.dropped_rows += int(missing_required.sum())
        df = df[~missing_required].copy()

    dupes = int(df["ticket_id"].duplicated().sum())
    if dupes:
        report.warnings.append(f"Dropped {dupes} duplicate ticket_id row(s)")
        report.dropped_rows += dupes
        df = df.drop_duplicates(subset="ticket_id", keep="first").copy()

    for col, domain in CATEGORICAL_DOMAINS.items():
        unknown = sorted(set(df[col].dropna().unique()) - domain)
        if unknown:
            report.warnings.append(
                f"Unexpected {col} value(s) present and kept as-is: {unknown}"
            )

    # Data-quality observations that matter for interpreting results.
    resolved_missing_time = int(
        ((df["status"] == "Resolved") & df["resolution_time_hrs"].isna()).sum()
    )
    if resolved_missing_time:
        report.warnings.append(
            f"{resolved_missing_time} Resolved ticket(s) have no resolution_time_hrs"
        )
    unresolved_with_time = int(
        ((df["status"] != "Resolved") & df["resolution_time_hrs"].notna()).sum()
    )
    if unresolved_with_time:
        report.warnings.append(
            f"{unresolved_with_time} unresolved ticket(s) have a resolution_time_hrs"
        )
    inverted = int((df["resolution_time_hrs"] < df["response_time_hrs"]).sum())
    if inverted:
        report.warnings.append(
            f"{inverted} ticket(s) have resolution_time_hrs < response_time_hrs "
            "(logically impossible; surfaced by the data_quality anomaly rule)"
        )

    df = df.sort_values("created_at").reset_index(drop=True)
    report.rows = len(df)
    report.null_counts = {c: int(df[c].isna().sum()) for c in EXPECTED_COLUMNS}
    report.min_created_at = df["created_at"].min()
    report.max_created_at = df["created_at"].max()

    for warning in report.warnings:
        logger.warning("data-quality: %s", warning)
    logger.info(
        "Loaded %d tickets from %s (%s -> %s)",
        report.rows,
        path.name,
        report.min_created_at,
        report.max_created_at,
    )
    return df, report
