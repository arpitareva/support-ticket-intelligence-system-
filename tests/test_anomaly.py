"""Anomaly detection rules and thresholds."""

from __future__ import annotations

from datetime import datetime

import pytest

from app.config import Settings
from app.database.database import TicketDatabase, build_database
from app.dependencies import AppState
from app.services.anomaly_service import AnomalyService
from app.utils.data_loader import load_tickets


def by_type(state: AppState, anomaly_type: str):
    return [
        a
        for a in state.anomaly_service.detect().anomalies
        if a.anomaly_type == anomaly_type
    ]


def test_reference_time_defaults_to_latest_ticket(state: AppState) -> None:
    """Ageing must not use the wall clock on a historical extract."""
    df = state.db.load_dataframe()
    reference, basis = state.anomaly_service.resolve_reference_time(df)

    assert reference == datetime(2024, 1, 8, 16, 0)
    assert "dataset" in basis


def test_reference_time_can_be_pinned_to_a_fixed_timestamp(
    settings: Settings,
) -> None:
    settings.anomaly_reference_time = "2024-01-02 00:00:00"
    df, _ = load_tickets(settings.csv_path)
    build_database(df, settings.sqlite_path)
    service = AnomalyService(TicketDatabase(settings.sqlite_path), settings)

    reference, basis = service.resolve_reference_time(service.db.load_dataframe())

    assert reference == datetime(2024, 1, 2, 0, 0)
    assert "configuration" in basis


def test_invalid_reference_time_falls_back_to_the_dataset(
    settings: Settings,
) -> None:
    settings.anomaly_reference_time = "not-a-timestamp"
    df, _ = load_tickets(settings.csv_path)
    build_database(df, settings.sqlite_path)
    service = AnomalyService(TicketDatabase(settings.sqlite_path), settings)

    reference, _ = service.resolve_reference_time(service.db.load_dataframe())

    assert reference == datetime(2024, 1, 8, 16, 0)


def test_aging_rule_flags_only_unresolved_high_priority(state: AppState) -> None:
    flagged = by_type(state, "unresolved_high_priority_aging")

    # Unresolved High/Critical in the fixture: TKT-002 (Critical, Escalated,
    # 174h old) and TKT-009 (High, Escalated, 25h old). TKT-005 and TKT-008 are
    # unresolved but only Medium priority.
    assert {a.ticket_id for a in flagged} == {"TKT-002", "TKT-009"}
    assert all(a.threshold == 24.0 for a in flagged)
    assert all(a.value > 24.0 for a in flagged)


def test_aging_severity_escalates_with_age_and_priority(state: AppState) -> None:
    flagged = {a.ticket_id: a for a in by_type(state, "unresolved_high_priority_aging")}

    assert flagged["TKT-002"].severity == "critical"  # Critical and > 7 days
    assert flagged["TKT-009"].severity == "low"  # High and only just past 24h


def test_aging_rule_respects_a_custom_threshold(settings: Settings) -> None:
    settings.aging_threshold_hours = 100.0
    df, _ = load_tickets(settings.csv_path)
    build_database(df, settings.sqlite_path)
    service = AnomalyService(TicketDatabase(settings.sqlite_path), settings)

    flagged = [
        a
        for a in service.detect().anomalies
        if a.anomaly_type == "unresolved_high_priority_aging"
    ]

    assert {a.ticket_id for a in flagged} == {"TKT-002"}


def test_long_resolution_needs_enough_samples(state: AppState) -> None:
    """With 6 resolved tickets a percentile is not defensible, so no flags."""
    assert by_type(state, "long_resolution") == []


def test_long_resolution_uses_the_95th_percentile(real_settings: Settings) -> None:
    df, _ = load_tickets(real_settings.csv_path)
    build_database(df, real_settings.sqlite_path)
    service = AnomalyService(TicketDatabase(real_settings.sqlite_path), real_settings)

    response = service.detect(types=["long_resolution"])
    thresholds = response.summary.thresholds

    assert thresholds["long_resolution_percentile"] == 95.0
    assert thresholds["long_resolution_sample_size"] == 327  # nulls excluded
    assert thresholds["long_resolution_hours"] == pytest.approx(60.12, abs=0.01)
    assert response.summary.total == 17
    assert all(a.value > thresholds["long_resolution_hours"] for a in response.anomalies)


def test_long_resolution_excludes_unresolved_tickets(real_settings: Settings) -> None:
    df, _ = load_tickets(real_settings.csv_path)
    build_database(df, real_settings.sqlite_path)
    service = AnomalyService(TicketDatabase(real_settings.sqlite_path), real_settings)

    flagged = service.detect(types=["long_resolution"]).anomalies
    resolved_ids = set(df.loc[df["status"] == "Resolved", "ticket_id"])

    assert {a.ticket_id for a in flagged} <= resolved_ids


def test_low_customer_rating_rule(state: AppState) -> None:
    flagged = {a.ticket_id: a for a in by_type(state, "low_customer_rating")}

    assert set(flagged) == {"TKT-006", "TKT-007"}  # ratings 2 and 1
    assert flagged["TKT-007"].severity == "high"  # rating 1
    assert flagged["TKT-006"].severity == "medium"  # rating 2


def test_data_quality_rule_flags_impossible_timings(state: AppState) -> None:
    flagged = by_type(state, "data_quality_inconsistent_timings")

    # TKT-007: first response at 4.0h, resolution recorded at 3.0h.
    assert [a.ticket_id for a in flagged] == ["TKT-007"]
    assert flagged[0].value == 1.0
    assert flagged[0].threshold == 0.0


def test_agent_outliers_are_entity_level_not_ticket_level(state: AppState) -> None:
    flagged = [
        a
        for a in state.anomaly_service.detect().anomalies
        if a.entity_type == "agent"
    ]

    assert flagged, "expected at least one agent-level outlier in the fixture"
    assert all(a.ticket_id is None for a in flagged)
    assert all(a.entity_id.startswith("AGT-") for a in flagged)


def test_agent_outliers_skip_agents_below_the_minimum_sample(
    settings: Settings,
) -> None:
    settings.min_tickets_for_agent_outlier = 99
    df, _ = load_tickets(settings.csv_path)
    build_database(df, settings.sqlite_path)
    service = AnomalyService(TicketDatabase(settings.sqlite_path), settings)

    flagged = [a for a in service.detect().anomalies if a.entity_type == "agent"]

    assert flagged == []


def test_every_anomaly_is_self_explanatory(state: AppState) -> None:
    """The required fields must all be populated on every flag."""
    for anomaly in state.anomaly_service.detect().anomalies:
        assert anomaly.entity_id
        assert anomaly.anomaly_type
        assert anomaly.severity in {"low", "medium", "high", "critical"}
        assert anomaly.metric
        assert isinstance(anomaly.value, float)
        assert isinstance(anomaly.threshold, float)
        assert anomaly.threshold_basis
        assert anomaly.reason.endswith(".")


def test_summary_tallies_match_the_anomaly_list(state: AppState) -> None:
    response = state.anomaly_service.detect()

    assert response.summary.total == len(response.anomalies)
    assert sum(response.summary.by_type.values()) == response.summary.total
    assert sum(response.summary.by_severity.values()) == response.summary.total


def test_type_filter_narrows_results(state: AppState) -> None:
    response = state.anomaly_service.detect(types=["low_customer_rating"])

    assert set(response.summary.by_type) == {"low_customer_rating"}


def test_detection_is_deterministic(state: AppState) -> None:
    first = state.anomaly_service.detect()
    second = state.anomaly_service.detect()

    assert [a.model_dump() for a in first.anomalies] == [
        a.model_dump() for a in second.anomalies
    ]


def test_real_dataset_anomaly_counts(real_settings: Settings) -> None:
    """Pins the headline figures for the shipped dataset."""
    df, _ = load_tickets(real_settings.csv_path)
    build_database(df, real_settings.sqlite_path)
    service = AnomalyService(TicketDatabase(real_settings.sqlite_path), real_settings)

    summary = service.detect().summary

    assert summary.reference_time == datetime(2024, 3, 30, 18, 6)
    assert summary.by_type["unresolved_high_priority_aging"] == 80
    assert summary.by_type["long_resolution"] == 17
    assert summary.by_type["low_customer_rating"] == 47
    assert summary.by_type["data_quality_inconsistent_timings"] == 28
