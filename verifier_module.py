import json
from typing import Any, Dict, List

from kg_module import KnowledgeGraphStore
from model_manager import ModelManager


VERIFIER_PROMPT_TEMPLATE = """You are a verification agent.

You are given:

1. A generated answer
2. Knowledge Graph evidence

RULES:

* Use ONLY the provided KG evidence for validation
* If KG contradicts answer → mark as contradiction
* If KG supports answer → mark as valid
* If KG missing data → fallback to reasoning

OUTPUT FORMAT (STRICT JSON):
{
"status": "valid" | "contradiction" | "uncertain",
"confidence": float,
"evidence": ["list of supporting triples"],
"feedback": "what is wrong or missing",
"source_of_truth": "KG" | "LLM" | "mixed"
}

DO NOT output anything outside JSON."""


class VerifierAgent:
    """Strict hybrid verifier using KG subgraph + LLM reasoning."""

    def __init__(self, model_manager: ModelManager, kg_store: KnowledgeGraphStore) -> None:
        self.model_manager = model_manager
        self.kg_store = kg_store

    async def run(self, answer: str, kg_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        relevant_context = kg_context
        prompt = (
            f"{VERIFIER_PROMPT_TEMPLATE}\n\n"
            f"Generated answer:\n{answer}\n\n"
            f"Knowledge Graph evidence:\n{json.dumps(relevant_context, ensure_ascii=False)}\n"
        )

        result = await self.model_manager.generate("verifier", prompt)
        parsed = self._parse_or_default(result.get("response", "{}"), relevant_context)

        # Enforce strict policy: confidence < 0.6 is failure.
        if parsed["confidence"] < 0.6 and parsed["status"] == "valid":
            parsed["status"] = "uncertain"
            parsed["feedback"] = "Confidence below 0.6; verification considered failed."

        if parsed["status"] == "contradiction" and not parsed["feedback"].strip():
            parsed["feedback"] = "KG evidence contradicts the generated answer."

        return parsed

    def _parse_or_default(self, raw: str, fallback_evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {
                "status": "uncertain",
                "confidence": 0.4,
                "evidence": fallback_evidence,
                "feedback": "Verifier output was not valid JSON.",
                "source_of_truth": "LLM" if not fallback_evidence else "mixed",
            }

        status = parsed.get("status", "uncertain")
        if status not in {"valid", "contradiction", "uncertain"}:
            status = "uncertain"

        confidence = float(parsed.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))

        evidence = parsed.get("evidence", fallback_evidence)
        if not isinstance(evidence, list):
            evidence = fallback_evidence

        feedback = str(parsed.get("feedback", "")).strip()
        source_of_truth = parsed.get("source_of_truth", "mixed")
        if source_of_truth not in {"KG", "LLM", "mixed"}:
            source_of_truth = "mixed"

        if status == "contradiction" and not feedback:
            feedback = "Detected contradiction with provided KG evidence."

        return {
            "status": status,
            "confidence": confidence,
            "evidence": evidence,
            "feedback": feedback,
            "source_of_truth": source_of_truth,
        }
