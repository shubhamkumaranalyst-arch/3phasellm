from typing import Any, Dict, Optional

from agents import LibrarianAgent, PerceiverAgent
from observability import SessionLogger
from verifier_module import VerifierAgent


class SupervisorOrchestrator:
    """Cyclic control loop over three agents with strict decision policy."""

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

        best_attempt: Dict[str, Any] = {
            "answer": "",
            "verification": {"status": "uncertain", "confidence": 0.0, "feedback": ""},
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
            state["knowledge_context"] = p_out.get("kg_context", [])
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
            self.logger.end_span(span_2, l_out)

            span_3 = self.logger.start_span(
                agent_name="VerifierAgent",
                iteration=state["iteration"],
                input_payload={
                    "current_answer": state["current_answer"],
                    "knowledge_context": state["knowledge_context"],
                },
            )
            v_out = await self.verifier.run(
                answer=state["current_answer"],
                kg_context=state["knowledge_context"],
            )
            state["verification"] = v_out
            self.logger.end_span(span_3, v_out)

            # Track best attempt by confidence.
            if float(v_out.get("confidence", 0.0)) >= float(best_attempt["verification"].get("confidence", 0.0)):
                best_attempt = {
                    "answer": state["current_answer"],
                    "verification": v_out,
                }

            status = v_out.get("status", "uncertain")
            confidence = float(v_out.get("confidence", 0.0))

            # ACCEPT only if valid and confidence >= 0.6
            if status == "valid" and confidence >= 0.6:
                return {
                    "decision": "ACCEPT",
                    "state": state,
                    "trace_id": self.logger.trace_id,
                }

            # Force retry for contradiction/uncertain OR low confidence.
            raw_feedback = str(v_out.get("feedback", "Please improve factual consistency.")).strip()
            feedback = f"Previous answer was incorrect because: {raw_feedback}"

        # Final fallback when retry limit is reached.
        state["current_answer"] = best_attempt["answer"]
        state["verification"] = best_attempt["verification"]
        warning = "Low confidence answer"
        return {
            "decision": "RETRY_LIMIT_REACHED",
            "warning": warning,
            "state": state,
            "trace_id": self.logger.trace_id,
        }
