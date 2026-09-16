
from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import httpx
from pydantic import ValidationError

from app.config import Settings
from app.schemas.query_plan import QueryPlan
from app.services.prompts import build_repair_prompt, build_system_prompt
from app.utils.errors import (
    LLMConfigurationError,
    LLMOutputError,
    LLMUnavailableError,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    name: str = "unknown"
    model: str | None = None

    @abstractmethod
    def complete_json(self, system: str, messages: list[dict[str, str]]) -> str:
        """Return the model's raw text response (expected to be JSON)."""

    def check_configuration(self) -> None:
        """Raise LLMConfigurationError if the provider cannot be used."""


class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.groq_model

    def check_configuration(self) -> None:
        if not self.settings.groq_api_key:
            raise LLMConfigurationError(
                "GROQ_API_KEY is not set",
                "Add GROQ_API_KEY to your .env (see .env.example), or set "
                "LLM_PROVIDER=ollama to use a local model.",
            )

    def complete_json(self, system: str, messages: list[dict[str, str]]) -> str:
        self.check_configuration()
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 900,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, *messages],
        }
        try:
            response = httpx.post(
                f"{self.settings.groq_base_url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.settings.groq_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.settings.llm_timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise LLMUnavailableError(
                "The language model timed out", str(exc)
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(
                "Could not reach the Groq API", str(exc)
            ) from exc

        if response.status_code == 401:
            raise LLMConfigurationError(
                "Groq rejected the API key", "Check GROQ_API_KEY in your .env."
            )
        if response.status_code == 429:
            raise LLMUnavailableError(
                "Groq rate limit reached", "Wait a moment and retry."
            )
        if response.status_code >= 400:
            raise LLMUnavailableError(
                f"Groq API error (HTTP {response.status_code})",
                response.text[:400],
            )
        try:
            return response.json()["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            raise LLMUnavailableError(
                "Unexpected response shape from Groq", str(exc)
            ) from exc


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.ollama_model

    def complete_json(self, system: str, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
            "messages": [{"role": "system", "content": system}, *messages],
        }
        try:
            response = httpx.post(
                f"{self.settings.ollama_base_url}/api/chat",
                json=payload,
                timeout=self.settings.llm_timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise LLMUnavailableError("The local model timed out", str(exc)) from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(
                "Could not reach the Ollama server",
                f"Is `ollama serve` running at {self.settings.ollama_base_url}? {exc}",
            ) from exc
        if response.status_code >= 400:
            raise LLMUnavailableError(
                f"Ollama error (HTTP {response.status_code})", response.text[:400]
            )
        try:
            return response.json()["message"]["content"] or ""
        except (KeyError, ValueError, TypeError) as exc:
            raise LLMUnavailableError(
                "Unexpected response shape from Ollama", str(exc)
            ) from exc


class RuleBasedProvider(LLMProvider):
    """Keyword planner used only when no model is available.

    It covers the common question shapes so the rest of the pipeline can be
    exercised offline. It is intentionally simple and will return
    ``intent: unsupported`` rather than guess.
    """

    name = "rule_based"
    model = "keyword-planner-v1"

    def __init__(self, settings: Settings, reference_date: datetime | None = None) -> None:
        self.settings = settings
        self.reference_date = reference_date

    def complete_json(self, system: str, messages: list[dict[str, str]]) -> str:
        question = messages[-1]["content"] if messages else ""
        return json.dumps(self._plan(question.lower()))

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _numbers(text: str) -> list[float]:
        return [float(m) for m in re.findall(r"\d+(?:\.\d+)?", text)]

    def _plan(self, q: str) -> dict[str, Any]:
        filters: list[dict[str, Any]] = []
        categories = [
            v.capitalize() for v in ("billing", "technical", "general") if v in q
        ]
        priorities = [
            v.capitalize()
            for v in ("critical", "high", "medium", "low")
            if re.search(rf"\b{v}\b", q)
        ]
        for field_name, values in (("category", categories), ("priority", priorities)):
            if len(values) == 1:
                filters.append({"field": field_name, "op": "eq", "value": values[0]})
            elif len(values) > 1 and not re.search(r"\bcompare\b|\bversus\b|\bvs\b", q):
                # "High and Critical tickets" is one population, not a comparison.
                filters.append({"field": field_name, "op": "in", "value": values})

        # "not resolved within N hours" is a duration breach, not a status.
        any_of: list[dict[str, Any]] = []
        breach = re.search(r"(?:within|under|in)\s+(\d+(?:\.\d+)?)\s*(?:hour|hr)", q)
        if breach and re.search(r"not resolved|unresolved|exceed|longer|over", q):
            any_of = [
                {
                    "field": "resolution_time_hrs",
                    "op": "gt",
                    "value": float(breach.group(1)),
                },
                {"field": "resolved", "op": "eq", "value": False},
            ]
        elif re.search(r"\bunresolved\b|\bnot resolved\b|\bstill (open|pending)\b", q):
            filters.append({"field": "resolved", "op": "eq", "value": False})
        elif re.search(r"\bopen\b", q):
            filters.append({"field": "status", "op": "eq", "value": "Open"})
        elif re.search(r"\bescalated\b", q):
            filters.append({"field": "status", "op": "eq", "value": "Escalated"})
        elif re.search(r"\bresolved\b", q):
            filters.append({"field": "resolved", "op": "eq", "value": True})

        metric: dict[str, Any] = {"agg": "count", "field": None}
        if "rating" in q or "satisfaction" in q:
            metric = {"agg": "avg", "field": "customer_rating"}
        elif "resolution time" in q or "resolve time" in q:
            metric = {"agg": "avg", "field": "resolution_time_hrs"}
        elif "response time" in q or "first response" in q:
            metric = {"agg": "avg", "field": "response_time_hrs"}

        superlative = bool(
            re.search(r"highest|lowest|most|least|best|worst|top|slowest|longest", q)
        )
        breakdown = bool(re.search(r"\bper\b|\beach\b|\bby (agent|category|priority)\b|breakdown", q))
        group_by = None
        if superlative or breakdown:
            if "agent" in q:
                group_by = "agent_id"
            elif "categor" in q and not categories:
                group_by = "category"
            elif "priorit" in q and not priorities:
                group_by = "priority"

        wants_lowest = bool(re.search(r"lowest|worst|least|fastest|quickest", q))
        direction = "asc" if wants_lowest else "desc"

        # "below-average X and above-average Y" is a relative comparison, not
        # a plain aggregate - answering it as one would be confidently wrong.
        relative = re.findall(r"(below|above)[- ]average\s+([a-z ]+?)(?:\s+and\b|[,.?]|$)", q)
        if relative and (group_by or "agent" in q or "categor" in q):
            metric_for = {
                "rating": {"agg": "avg", "field": "customer_rating"},
                "satisfaction": {"agg": "avg", "field": "customer_rating"},
                "resolution": {"agg": "avg", "field": "resolution_time_hrs"},
                "response": {"agg": "avg", "field": "response_time_hrs"},
            }
            conditions = []
            for direction_word, phrase in relative:
                for keyword, spec in metric_for.items():
                    if keyword in phrase:
                        conditions.append(
                            {"metric": spec, "direction": direction_word}
                        )
                        break
            if conditions:
                return {
                    "intent": "relative_outliers",
                    "filters": filters,
                    "group_by": group_by
                    or ("agent_id" if "agent" in q else "category"),
                    "metric": {"agg": "count", "field": None},
                    "relative_conditions": conditions,
                    "having_min_count": 5,
                    "reason": "Rule-based planner: groups compared to the overall average.",
                }

        # Two-part question: a superlative picks the group AND the user wants
        # the underlying rows ("...and what are those ticket IDs?"). This must
        # be tested before the plain list branch, whose keywords it shares.
        wants_rows = bool(
            re.search(
                r"ticket ids|ticket id|which tickets|what are those|list them|"
                r"show (me )?(those|them|the tickets)",
                q,
            )
        )
        if group_by and superlative and wants_rows:
            return {
                "intent": "group_detail",
                "filters": filters,
                "any_of": any_of,
                "group_by": group_by,
                "metric": metric,
                "sort": {"by": "metric", "direction": direction},
                "limit": 1,
                "having_min_count": int(self._numbers(q)[0])
                if "at least" in q and self._numbers(q)
                else None,
                "detail": {
                    "sort": {"by": "created_at", "direction": "asc"},
                    "limit": 50,
                },
                "reason": (
                    "Rule-based planner: ranked groups, then listed the "
                    "winning group's tickets."
                ),
            }

        if re.search(r"\bshow me\b|\blist\b|\bwhich tickets\b|ticket ids\b", q):
            return {
                "intent": "list",
                "filters": filters,
                "any_of": any_of,
                "sort": {"by": "resolution_time_hrs", "direction": "desc"},
                "limit": 50,
                "reason": "Rule-based planner: list intent.",
            }

        if "compare" in q and group_by:
            arms = [
                {"label": f["value"], "filters": [f]}
                for f in filters
                if f["field"] in ("priority", "category", "status")
            ]
            if len(arms) >= 2:
                return {
                    "intent": "compare",
                    "metric": metric,
                    "compare": arms,
                }

        if group_by and superlative:
            return {
                "intent": "group_aggregate",
                "filters": filters,
                "group_by": group_by,
                "metric": metric,
                "sort": {"by": "metric", "direction": direction},
                "limit": 1,
                "having_min_count": int(self._numbers(q)[0])
                if "at least" in q and self._numbers(q)
                else None,
            }
        if group_by:
            return {
                "intent": "group_aggregate",
                "filters": filters,
                "group_by": group_by,
                "metric": metric,
                "sort": {"by": "metric", "direction": "desc"},
            }
        if metric["agg"] != "count":
            return {
                "intent": "aggregate",
                "filters": filters,
                "any_of": any_of,
                "metric": metric,
            }
        if re.search(r"how many|count|number of", q):
            return {"intent": "count", "filters": filters, "any_of": any_of}
        return {
            "intent": "unsupported",
            "reason": (
                "The offline rule-based planner could not interpret this "
                "question. Configure GROQ_API_KEY or Ollama for full "
                "natural-language understanding."
            ),
        }


def build_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "groq":
        return GroqProvider(settings)
    if settings.llm_provider == "ollama":
        return OllamaProvider(settings)
    return RuleBasedProvider(settings)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(raw: str) -> dict[str, Any]:
    """Pull a JSON object out of a model response.

    Models occasionally wrap JSON in prose or markdown fences even when asked
    not to, so the first attempt is a plain parse and the fallback is the
    outermost brace-delimited block.
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(text)
        if not match:
            raise LLMOutputError(
                "The language model did not return JSON",
                f"Raw response: {text[:300]}",
            )
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise LLMOutputError(
                "The language model returned malformed JSON", str(exc)
            ) from exc
    if not isinstance(parsed, dict):
        raise LLMOutputError(
            "The language model returned JSON that is not an object",
            f"Got {type(parsed).__name__}.",
        )
    return parsed


class LLMQueryPlanner:
    """Turns a question into a validated ``QueryPlan``."""

    def __init__(
        self,
        provider: LLMProvider,
        settings: Settings,
        min_date: datetime | None = None,
        max_date: datetime | None = None,
    ) -> None:
        self.provider = provider
        self.settings = settings
        self.system_prompt = build_system_prompt(min_date, max_date)

    @property
    def provider_name(self) -> str:
        return self.provider.name

    @property
    def model_name(self) -> str | None:
        return self.provider.model

    def plan(self, question: str) -> tuple[QueryPlan, dict[str, Any]]:
        messages: list[dict[str, str]] = [{"role": "user", "content": question}]
        attempts = 0
        last_error: str | None = None
        raw = ""
        started = time.perf_counter()

        while attempts <= self.settings.llm_max_retries:
            raw = self.provider.complete_json(self.system_prompt, messages)
            try:
                payload = extract_json(raw)
                payload = _normalise_payload(payload)
                plan = QueryPlan.model_validate(payload)
                meta = {
                    "llm_latency_ms": int((time.perf_counter() - started) * 1000),
                    "llm_repair_attempts": attempts,
                }
                return plan, meta
            except (LLMOutputError, ValidationError) as exc:
                last_error = (
                    exc.message if isinstance(exc, LLMOutputError) else str(exc)
                )
                logger.warning(
                    "Invalid query plan (attempt %d): %s", attempts + 1, last_error
                )
                attempts += 1
                if attempts > self.settings.llm_max_retries:
                    break
                messages = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": raw[:1500]},
                    {"role": "user", "content": build_repair_prompt(raw, last_error)},
                ]

        raise LLMOutputError(
            "The language model could not produce a valid query plan",
            f"Last validation error: {last_error}",
        )


def _normalise_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Tolerate harmless shape variations before strict validation.

    Cheaper and more predictable than a retry: drop nulls that Pydantic would
    reject as explicit ``None``, unwrap a single-key envelope, and accept
    ``group_by``/``metric`` given as bare strings.
    """
    if len(payload) == 1:
        only_value = next(iter(payload.values()))
        if isinstance(only_value, dict) and "intent" in only_value:
            payload = only_value

    payload = {k: v for k, v in payload.items() if v is not None}

    metric = payload.get("metric")
    if isinstance(metric, str):
        payload["metric"] = {"agg": metric}
    if isinstance(payload.get("metric"), dict):
        payload["metric"] = {
            k: v for k, v in payload["metric"].items() if v is not None
        }
    for key in ("secondary_metrics",):
        if isinstance(payload.get(key), list):
            payload[key] = [
                {k: v for k, v in m.items() if v is not None}
                for m in payload[key]
                if isinstance(m, dict)
            ]
    if isinstance(payload.get("relative_conditions"), list):
        for cond in payload["relative_conditions"]:
            if isinstance(cond, dict) and isinstance(cond.get("metric"), dict):
                cond["metric"] = {
                    k: v for k, v in cond["metric"].items() if v is not None
                }
    if isinstance(payload.get("sort"), dict):
        payload["sort"] = {k: v for k, v in payload["sort"].items() if v is not None}
    if isinstance(payload.get("time_range"), dict):
        payload["time_range"] = {
            k: v for k, v in payload["time_range"].items() if v is not None
        }
    for key in ("filters", "any_of"):
        if isinstance(payload.get(key), list):
            payload[key] = [f for f in payload[key] if isinstance(f, dict)]
    return payload


__all__ = [
    "LLMProvider",
    "GroqProvider",
    "OllamaProvider",
    "RuleBasedProvider",
    "LLMQueryPlanner",
    "build_provider",
    "extract_json",
]

