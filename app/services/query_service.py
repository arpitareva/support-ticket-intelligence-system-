
from __future__ import annotations

import logging

from app.config import Settings
from app.query.engine import QueryEngine
from app.schemas.query_plan import Intent, QueryPlan
from app.schemas.response import ExecutionMeta, QueryResponse
from app.services.answer_formatter import describe_plan, format_answer
from app.services.llm_service import LLMQueryPlanner
from app.utils.errors import UnsupportedQuestionError

logger = logging.getLogger(__name__)


class QueryService:
    def __init__(
        self,
        planner: LLMQueryPlanner,
        engine: QueryEngine,
        settings: Settings,
    ) -> None:
        self.planner = planner
        self.engine = engine
        self.settings = settings

    def answer(self, question: str) -> QueryResponse:
        plan, llm_meta = self.planner.plan(question)
        logger.info(
            "planned question=%r intent=%s", question[:120], plan.intent.value
        )

        if plan.intent is Intent.UNSUPPORTED:
            raise UnsupportedQuestionError(
                "That question cannot be answered from this dataset",
                plan.reason,
            )

        result = self.engine.execute(plan)
        answer = format_answer(plan, result)
        if result.notes:
            answer = f"{answer} " + " ".join(result.notes)

        return QueryResponse(
            question=question,
            answer=answer,
            query_plan=plan,
            plan_explanation=describe_plan(plan),
            results=result.rows,
            groups=result.group_rows,
            result_type=result.result_type,
            execution=ExecutionMeta(
                llm_provider=self.planner.provider_name,
                llm_model=self.planner.model_name,
                llm_latency_ms=llm_meta.get("llm_latency_ms"),
                llm_repair_attempts=llm_meta.get("llm_repair_attempts", 0),
                query_latency_ms=result.latency_ms,
                rows_returned=len(result.rows),
                row_limit_applied=result.row_limit_applied,
                sql=" | ".join(result.sql) if result.sql else None,
            ),
        )

    def run_plan(self, plan: QueryPlan, question: str = "(direct plan)") -> QueryResponse:
        """Execute a hand-written plan. Used by tests and debugging tools."""
        result = self.engine.execute(plan)
        return QueryResponse(
            question=question,
            answer=format_answer(plan, result),
            query_plan=plan,
            plan_explanation=describe_plan(plan),
            results=result.rows,
            groups=result.group_rows,
            result_type=result.result_type,
            execution=ExecutionMeta(
                llm_provider="none",
                query_latency_ms=result.latency_ms,
                rows_returned=len(result.rows),
                sql=" | ".join(result.sql) if result.sql else None,
            ),
        )
