"""CSV ingestion, schema validation and database initialisation."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.database.database import TicketDatabase, build_database
from app.utils.data_loader import load_tickets
from app.utils.errors import DataLoadError

HEADER = (
    "ticket_id,created_at,category,priority,status,response_time_hrs,"
    "resolution_time_hrs,agent_id,customer_rating,issue_summary\n"
)


def test_loads_expected_rows_and_dtypes(sample_csv: Path) -> None:
    df, report = load_tickets(sample_csv)

    assert report.rows == 10
    assert report.dropped_rows == 0
    assert str(df["created_at"].dtype).startswith("datetime64")
    assert df["response_time_hrs"].dtype.kind == "f"
    assert df["customer_rating"].dtype.kind == "f"


def test_nulls_only_on_unresolved_tickets(sample_csv: Path) -> None:
    df, report = load_tickets(sample_csv)

    assert report.null_counts["resolution_time_hrs"] == 4
    assert report.null_counts["customer_rating"] == 4
    assert report.null_counts["ticket_id"] == 0
    unresolved = df[df["status"] != "Resolved"]
    assert unresolved["resolution_time_hrs"].isna().all()


def test_missing_file_raises() -> None:
    with pytest.raises(DataLoadError, match="not found"):
        load_tickets("/nonexistent/support_tickets.csv")


def test_invalid_schema_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("ticket_id,created_at,category\nTKT-001,2024-01-01 09:00,Billing\n")

    with pytest.raises(DataLoadError, match="schema is invalid") as exc_info:
        load_tickets(path)
    assert "Missing required column" in (exc_info.value.detail or "")


def test_empty_dataset_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text(HEADER)

    with pytest.raises(DataLoadError, match="no data rows"):
        load_tickets(path)


def test_completely_empty_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "blank.csv"
    path.write_text("")

    with pytest.raises(DataLoadError):
        load_tickets(path)


def test_invalid_dates_are_dropped_with_a_warning(tmp_path: Path) -> None:
    path = tmp_path / "dates.csv"
    path.write_text(
        HEADER
        + "TKT-001,2024-01-01 09:00,Billing,High,Resolved,0.5,2.0,AGT-01,5,ok\n"
        + "TKT-002,not-a-date,Billing,High,Resolved,0.5,2.0,AGT-01,5,bad date\n"
    )

    df, report = load_tickets(path)

    assert report.rows == 1
    assert report.dropped_rows == 1
    assert any("created_at" in w for w in report.warnings)
    assert df["ticket_id"].tolist() == ["TKT-001"]


def test_all_invalid_dates_raises(tmp_path: Path) -> None:
    path = tmp_path / "alldates.csv"
    path.write_text(
        HEADER + "TKT-001,never,Billing,High,Resolved,0.5,2.0,AGT-01,5,bad\n"
    )

    with pytest.raises(DataLoadError, match="valid created_at"):
        load_tickets(path)


def test_duplicate_ticket_ids_are_deduplicated(tmp_path: Path) -> None:
    path = tmp_path / "dupes.csv"
    row = "TKT-001,2024-01-01 09:00,Billing,High,Resolved,0.5,2.0,AGT-01,5,ok\n"
    path.write_text(HEADER + row + row)

    _, report = load_tickets(path)

    assert report.rows == 1
    assert any("duplicate" in w for w in report.warnings)


def test_non_numeric_metrics_are_coerced_to_null(tmp_path: Path) -> None:
    path = tmp_path / "coerce.csv"
    path.write_text(
        HEADER + "TKT-001,2024-01-01 09:00,Billing,High,Resolved,0.5,fast,AGT-01,5,ok\n"
    )

    df, report = load_tickets(path)

    assert df["resolution_time_hrs"].isna().all()
    assert any("non-numeric" in w for w in report.warnings)


def test_inconsistent_timings_are_reported(sample_csv: Path) -> None:
    # TKT-007 responds at 4.0h but records resolution at 3.0h.
    _, report = load_tickets(sample_csv)

    assert any("resolution_time_hrs < response_time_hrs" in w for w in report.warnings)


def test_database_initialisation(sample_csv: Path, tmp_path: Path) -> None:
    df, _ = load_tickets(sample_csv)
    db_path = tmp_path / "t.db"

    inserted = build_database(df, db_path)
    db = TicketDatabase(db_path)

    assert inserted == 10
    assert db.count_tickets() == 10
    lo, hi = db.date_range()
    assert lo is not None and hi is not None
    assert lo.strftime("%Y-%m-%d") == "2024-01-01"
    assert hi.strftime("%Y-%m-%d") == "2024-01-08"


def test_derived_resolved_column_matches_status(sample_csv: Path, tmp_path: Path) -> None:
    df, _ = load_tickets(sample_csv)
    db_path = tmp_path / "t.db"
    build_database(df, db_path)
    db = TicketDatabase(db_path)

    rows = db.fetch_all("SELECT status, resolved FROM tickets")

    assert all(row["resolved"] == (row["status"] == "Resolved") for row in rows)


def test_connection_is_read_only(sample_csv: Path, tmp_path: Path) -> None:
    from app.utils.errors import DatabaseError

    df, _ = load_tickets(sample_csv)
    db_path = tmp_path / "t.db"
    build_database(df, db_path)
    db = TicketDatabase(db_path)

    with pytest.raises(DatabaseError):
        with db.connect() as conn:
            conn.execute("DELETE FROM tickets")
    assert db.count_tickets() == 10


def test_real_dataset_loads_cleanly(real_settings: Settings) -> None:
    """Guards against schema drift in the shipped CSV."""
    df, report = load_tickets(real_settings.csv_path)

    assert report.rows == 500
    assert report.dropped_rows == 0
    assert set(df["category"].unique()) == {"Billing", "Technical", "General"}
    assert set(df["status"].unique()) == {"Open", "Resolved", "Escalated"}
    assert report.null_counts["resolution_time_hrs"] == report.null_counts[
        "customer_rating"
    ]
