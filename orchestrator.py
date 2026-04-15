from typing import Any, Dict, Optional

from agents import LibrarianAgent, PerceiverAgent
from observability import SessionLogger
from verifier_module import VerifierAgent


class SupervisorOrchestrator:
    """Cyclic control loop over three agents with shared mutable state."""

    def __init__(
        self,
        perceiver: PerceiverAgent,
        librarian: LibrarianAgent,
        verifier: VerifierAgent,
        logger: SessionLogger,
        max_retries: int = 3,
    ) -> None:
        self.perceiver = perceiver
        self.librarian = librarian
        self.verifier = verifier
        self.logger = logger
        self.max_retries = max_retries

    async def run(
        self,
        user_query: str,
        file_path: Optional[str] = None,
        think: bool = False,
        research: bool = False,
    ) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "user_query": user_query,
            "current_answer": "",
            "knowledge_context": [],
            "verification": {},
            "iteration": 0,
        }

        feedback: Optional[str] = None
        while state["iteration"] < self.max_retries:
            state["iteration"] += 1

            span_1 = self.logger.start_span(
                agent_name="PerceiverAgent",
                iteration=state["iteration"],
                input_payload={
                    "user_query": state["user_query"],
                    "file_path": file_path,
                    "feedback": feedback,
                    "think": think,
                    "research": research,
                },
            )
            p_out = await self.perceiver.run(
                user_query=state["user_query"],
                file_path=file_path,
                think=think,
                research=research,
                feedback=feedback,
            )
            state["current_answer"] = p_out.get("answer", "")
            self.logger.end_span(span_1, p_out)

            span_2 = self.logger.start_span(
                agent_name="LibrarianAgent",
                iteration=state["iteration"],
                input_payload={
                    "user_query": state["user_query"],
                    "current_answer": state["current_answer"],
                    "file_path": file_path,
                },
            )
            l_out = await self.librarian.run(
                user_query=state["user_query"],
                answer=state["current_answer"],
                source_file=file_path,
            )
            state["knowledge_context"] = l_out.get("triples", [])
            self.logger.end_span(span_2, l_out)

            span_3 = self.logger.start_span(
                agent_name="VerifierAgent",
                iteration=state["iteration"],
                input_payload={
                    "current_answer": state["current_answer"],
                    "knowledge_context": state["knowledge_context"],
                },
            )
            v_out = await self.verifier.run(state["current_answer"])
            state["verification"] = v_out
            self.logger.end_span(span_3, v_out)

            status = v_out.get("status", "uncertain")
            if status == "valid":
                return {
                    "decision": "ACCEPT",
                    "state": state,
                    "trace_id": self.logger.trace_id,
                }

            if status in {"contradiction", "uncertain"}:
                feedback = v_out.get("feedback", "Please improve factual consistency.")
                continue

        return {
            "decision": "RETRY_LIMIT_REACHED",
            "state": state,
            "trace_id": self.logger.trace_id,
        }
