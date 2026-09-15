"""SQLite access.

Two responsibilities:

1. ``build_database`` - one-time load of the validated DataFrame into SQLite.
2. ``TicketDatabase.connect`` - hand out **read-only** connections for every
   request path. The connection is opened with ``mode=ro`` and additionally
   pinned with ``PRAGMA query_only``, so even a bug in the query builder
   cannot mutate or drop data.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from app.database.models import (
    COLUMNS,
    CREATE_INDEXES_SQL,
    CREATE_TABLE_SQL,
    TABLE_NAME,
)
from app.utils.errors import DatabaseError

logger = logging.getLogger(__name__)

DATETIME_FMT = "%Y-%m-%d %H:%M:%S"


def build_database(df: pd.DataFrame, sqlite_path: str | Path) -> int:
    """(Re)create the SQLite database from a validated DataFrame."""
    path = Path(sqlite_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out = df.copy()
    out["created_at"] = pd.to_datetime(out["created_at"]).dt.strftime(DATETIME_FMT)
    out["resolved"] = (out["status"] == "Resolved").astype(int)
    out = out[COLUMNS]

    try:
        with sqlite3.connect(path) as conn:
            conn.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
            conn.execute(CREATE_TABLE_SQL)
            for stmt in CREATE_INDEXES_SQL:
                conn.execute(stmt)
            conn.executemany(
                f"INSERT INTO {TABLE_NAME} ({', '.join(COLUMNS)}) "
                f"VALUES ({', '.join(['?'] * len(COLUMNS))})",
                out.itertuples(index=False, name=None),
            )
            conn.commit()
            count = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
    except sqlite3.Error as exc:  # pragma: no cover - environment failure
        raise DatabaseError(f"Failed to build SQLite database at {path}", str(exc)) from exc

    logger.info("Built SQLite database at %s with %d rows", path, count)
    return int(count)


class TicketDatabase:
    """Thin read-only accessor around the tickets table."""

    def __init__(self, sqlite_path: str | Path) -> None:
        self.path = Path(sqlite_path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        if not self.path.exists():
            raise DatabaseError(
                "Ticket database is not initialised",
                f"No SQLite file at {self.path}. Restart the API to rebuild it.",
            )
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only = ON")
            yield conn
        except sqlite3.Error as exc:
            raise DatabaseError("Database query failed", str(exc)) from exc
        finally:
            if conn is not None:
                conn.close()

    def fetch_all(self, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(sql, params or []).fetchall()
        return [dict(r) for r in rows]

    def fetch_one(self, sql: str, params: list[Any] | None = None) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(sql, params or []).fetchone()
        return dict(row) if row else None

    def count_tickets(self) -> int:
        row = self.fetch_one(f"SELECT COUNT(*) AS n FROM {TABLE_NAME}")
        return int(row["n"]) if row else 0

    def date_range(self) -> tuple[datetime | None, datetime | None]:
        row = self.fetch_one(
            f"SELECT MIN(created_at) AS lo, MAX(created_at) AS hi FROM {TABLE_NAME}"
        )
        if not row or not row["lo"]:
            return None, None
        return (
            datetime.strptime(row["lo"], DATETIME_FMT),
            datetime.strptime(row["hi"], DATETIME_FMT),
        )

    def load_dataframe(self) -> pd.DataFrame:
        """Full table as a DataFrame - used by the anomaly engine, which needs
        distribution-wide statistics (percentiles, standard deviations)."""
        with self.connect() as conn:
            df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
        if df.empty:
            raise DatabaseError(
                "Ticket table is empty", "Reload the CSV and restart the API."
            )
        df["created_at"] = pd.to_datetime(df["created_at"])
        return df
