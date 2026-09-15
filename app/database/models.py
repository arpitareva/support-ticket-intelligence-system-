"""Physical schema for the tickets table.

Kept as plain DDL (no ORM): the application has exactly one table and never
performs writes at request time, so an ORM would add a dependency and an
indirection layer without removing any work. See README - "Why no ORM".
"""

from __future__ import annotations

TABLE_NAME = "tickets"

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    ticket_id           TEXT PRIMARY KEY,
    created_at          TEXT NOT NULL,          -- ISO-8601 'YYYY-MM-DD HH:MM:SS'
    category            TEXT NOT NULL,
    priority            TEXT NOT NULL,
    status              TEXT NOT NULL,
    response_time_hrs   REAL,
    resolution_time_hrs REAL,                   -- NULL when unresolved
    agent_id            TEXT NOT NULL,
    customer_rating     REAL,                   -- NULL when unresolved
    issue_summary       TEXT NOT NULL,
    resolved            INTEGER NOT NULL        -- derived: status = 'Resolved'
);
"""

CREATE_INDEXES_SQL = [
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_status ON {TABLE_NAME}(status);",
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_priority ON {TABLE_NAME}(priority);",
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_category ON {TABLE_NAME}(category);",
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_agent ON {TABLE_NAME}(agent_id);",
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_created ON {TABLE_NAME}(created_at);",
]

COLUMNS = [
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
    "resolved",
]
