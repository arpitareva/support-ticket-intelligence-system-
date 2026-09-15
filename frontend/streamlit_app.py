from __future__ import annotations

import os
from typing import Any

import httpx
import pandas as pd
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 60.0

SAMPLE_QUESTIONS = [
    "How many tickets are currently open?",
    "Which agent resolved the most tickets?",
    "Show me all Critical tickets not resolved within 12 hours.",
    "What is the average customer rating for Technical category tickets?",
    "Which category has the highest average resolution time?",
    "Compare the average resolution time of High and Critical tickets.",
    "Which agent has the lowest average customer rating among agents who have resolved at least 5 tickets?",
    "Which agents have both a below-average rating and above-average resolution time?",
    "Among unresolved High and Critical tickets, which agent currently has the most tickets, and what are those ticket IDs?",
]

st.set_page_config(
    page_title="Support Ticket Intelligence",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .block-container {
          padding-top: 2.2rem;
          max-width: 1200px;
      }

      div[data-testid="stMetricValue"] {
          font-size: 1.6rem;
      }

      .answer-box {
          border-left: 4px solid #2f4f6f;
          background: rgba(47, 79, 111, 0.06);
          padding: 0.9rem 1.1rem;
          border-radius: 4px;
          font-size: 1.02rem;
          line-height: 1.55;
      }

      /* Increase tab text size */
      button[data-baseweb="tab"] {
          font-size: 25px !important;
          font-weight: 700 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)
# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def api_get(path: str, params: dict[str, Any] | None = None) -> tuple[Any, str | None]:
    try:
        response = httpx.get(f"{API_BASE_URL}{path}", params=params, timeout=TIMEOUT)
    except httpx.HTTPError as exc:
        return None, (
            f"Cannot reach the API at {API_BASE_URL}. Start it with "
            f"`uvicorn app.main:app --reload`. ({exc})"
        )
    return _unwrap(response)


def api_post(path: str, payload: dict[str, Any]) -> tuple[Any, str | None]:
    try:
        response = httpx.post(f"{API_BASE_URL}{path}", json=payload, timeout=TIMEOUT)
    except httpx.HTTPError as exc:
        return None, (
            f"Cannot reach the API at {API_BASE_URL}. Start it with "
            f"`uvicorn app.main:app --reload`. ({exc})"
        )
    return _unwrap(response)


def _unwrap(response: httpx.Response) -> tuple[Any, str | None]:
    try:
        body = response.json()
    except ValueError:
        return None, f"HTTP {response.status_code}: {response.text[:300]}"
    if response.status_code >= 400:
        message = body.get("message") or body.get("detail") or "Request failed"
        detail = body.get("detail")
        if detail and detail != message:
            return None, f"{message} - {detail}"
        return None, str(message)
    return body, None


@st.cache_data(ttl=60, show_spinner=False)
def cached_get(path: str) -> tuple[Any, str | None]:
    return api_get(path)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

health, health_error = cached_get("/health")

with st.sidebar:
    st.subheader("Service")
    st.caption(f"API: `{API_BASE_URL}`")
    if health_error:
        st.error(health_error)
    else:
        badge = {"healthy": "🟢", "degraded": "🟡", "unhealthy": "🔴"}.get(
            health["status"], "⚪"
        )
        st.write(f"{badge} **{health['status']}**")
        st.write(f"Tickets loaded: **{health['tickets_loaded']}**")
        st.write(f"LLM: **{health['llm_provider']}**")
        if not health["llm_configured"]:
            st.warning(
                "No LLM credentials detected. Ask AI will return an error until "
                "GROQ_API_KEY is set (or LLM_PROVIDER is switched to ollama)."
            )
        if health["llm_provider"] == "rule_based":
            st.info(
                "Running the offline rule-based planner. It covers common "
                "question shapes only - configure an LLM for full coverage."
            )
        rng = health.get("dataset_range") or {}
        if rng.get("start"):
            st.caption(f"Data: {rng['start'][:10]} to {rng['end'][:10]}")
        if health.get("anomaly_reference_time"):
            st.caption(f"Anomaly reference: {health['anomaly_reference_time'][:16]}")
    if st.button("Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.title("Support Ticket Intelligence")
# st.caption(
#     "Natural-language questions are translated into a validated query plan and "
#     "executed in SQL. Every number below is computed from the dataset, not "
#     "generated by a language model."
# )

tab_ask, tab_anomalies,tab_dashboard = st.tabs(["Ask AI", "Anomalies","Dashboard"])


# ---------------------------------------------------------------------------
# Ask AI
# ---------------------------------------------------------------------------


def set_question(sample: str):
    st.session_state.question = sample


with tab_ask:
    st.text_input(
        "Ask a question about your support tickets...",
        key="question",
        placeholder="e.g. How many High priority tickets are unresolved?",
    )

    submitted = st.button("Ask", type="primary")

    with st.expander("Example questions"):
        example_columns = st.columns(2)

        for index, sample in enumerate(SAMPLE_QUESTIONS):
            example_columns[index % 2].button(
                sample,
                key=f"sample-{index}",
                use_container_width=True,
                on_click=set_question,
                args=(sample,),
            )

    if submitted and not st.session_state.question.strip():
        st.warning("Enter a question first.")
    elif submitted:
        with st.spinner("Planning the query and running it against SQLite..."):
            payload, error = api_post(
                "/query", {"question": st.session_state.question.strip()}
            )
        if error:
            st.error(error)
            st.caption(
                "The system reports failures instead of guessing an answer. "
                "Rephrase the question, or check that the LLM provider is "
                "reachable."
            )
        else:
            st.markdown(
                f"<div class='answer-box'>{payload['answer']}</div>",
                unsafe_allow_html=True,
            )
            meta = payload["execution"]
            st.caption(
                f"interpretation: {payload['plan_explanation']}  \n"
                f"provider: {meta['llm_provider']}"
                f"{' / ' + meta['llm_model'] if meta.get('llm_model') else ''}"
                f" | plan {meta.get('llm_latency_ms') or 0} ms"
                f" | query {meta.get('query_latency_ms') or 0} ms"
                f" | rows {meta['rows_returned']}"
            )

            # Two-stage answers carry the ranking that selected the entity as
            # well as its rows; showing both makes the result checkable.
            if payload.get("groups"):
                st.caption("Ranking that selected the result")
                st.dataframe(
                    pd.DataFrame(payload["groups"]),
                    use_container_width=True,
                    hide_index=True,
                )

            if payload["results"]:
                if payload.get("groups"):
                    st.caption("Matching tickets")
                st.dataframe(
                    pd.DataFrame(payload["results"]),
                    use_container_width=True,
                    hide_index=True,
                )

            with st.expander("Structured query plan and SQL"):
                st.json(payload["query_plan"])
                if meta.get("sql"):
                    st.code(meta["sql"].replace(" | ", ";\n"), language="sql")


# ---------------------------------------------------------------------------
# Anomalies
# ---------------------------------------------------------------------------

with tab_anomalies:
    data, error = cached_get("/anomalies")
    if error:
        st.error(error)
    else:
        summary = data["summary"]
        columns = st.columns(5)
        columns[0].metric("Total anomalies", summary["total"])
        for column, severity in zip(
            columns[1:], ["critical", "high", "medium", "low"]
        ):
            column.metric(
                severity.capitalize(), summary["by_severity"].get(severity, 0)
            )

        st.caption(
            f"Reference time: {summary['reference_time']} "
            f"({summary['thresholds'].get('reference_time_basis', 'n/a')})"
        )

        anomalies = pd.DataFrame(data["anomalies"])
        if anomalies.empty:
            st.success("No anomalies detected.")
        else:
            filters = st.columns(2)
            types = filters[0].multiselect(
                "Anomaly type",
                sorted(summary["by_type"]),
                default=sorted(summary["by_type"]),
                format_func=lambda t: f"{t} ({summary['by_type'][t]})",
            )
            severities = filters[1].multiselect(
                "Severity",
                ["critical", "high", "medium", "low"],
                default=["critical", "high", "medium", "low"],
            )
            view = anomalies[
                anomalies["anomaly_type"].isin(types)
                & anomalies["severity"].isin(severities)
            ]

            st.write(f"Showing **{len(view)}** of {len(anomalies)} anomalies.")
            st.dataframe(
                view[
                    [
                        "ticket_id",
                        "entity_type",
                        "entity_id",
                        "anomaly_type",
                        "severity",
                        "metric",
                        "value",
                        "threshold",
                        "reason",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "reason": st.column_config.TextColumn("reason", width="large")
                },
            )

            with st.expander("Detection thresholds used"):
                st.json(summary["thresholds"])
            with st.expander("Counts by type"):
                st.bar_chart(pd.Series(summary["by_type"], name="anomalies"))

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

with tab_dashboard:
    stats, stats_error = cached_get("/stats")
    if stats_error:
        st.error(stats_error)
    else:
        by_status = stats["by_status"]
        row1 = st.columns(4)
        row1[0].metric("Total tickets", stats["total_tickets"])
        row1[1].metric("Open", by_status.get("Open", 0))
        row1[2].metric("Resolved", by_status.get("Resolved", 0))
        row1[3].metric("Escalated", by_status.get("Escalated", 0))

        row2 = st.columns(4)
        by_priority = stats["by_priority"]
        row2[0].metric("Critical", by_priority.get("Critical", 0))
        row2[1].metric("High", by_priority.get("High", 0))
        row2[2].metric("Unresolved", stats["unresolved_tickets"])
        row2[3].metric("Anomalies flagged", stats["anomaly_count"])

        row3 = st.columns(4)
        row3[0].metric("Avg rating", stats["avg_customer_rating"] or 0.0)
        row3[1].metric("Avg response (h)", stats["avg_response_time_hrs"] or 0.0)
        row3[2].metric("Avg resolution (h)", stats["avg_resolution_time_hrs"] or 0.0)
        row3[3].metric("Median resolution (h)", stats["median_resolution_time_hrs"] or 0.0)

        st.divider()
        charts = st.columns(3)
        for column, (title, data) in zip(
            charts,
            [
                ("By status", stats["by_status"]),
                ("By category", stats["by_category"]),
                ("By priority", stats["by_priority"]),
            ],
        ):
            with column:
                st.caption(title)
                st.bar_chart(
                    pd.Series(data, name="tickets").sort_values(ascending=False),
                    height=240,
                )

        st.caption(
            f"{stats['rated_tickets']} of {stats['total_tickets']} tickets carry a "
            "customer rating; ratings and resolution times are null for "
            "unresolved tickets and are excluded from the averages above."
        )

