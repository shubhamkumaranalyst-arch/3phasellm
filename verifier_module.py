import json
import re
from typing import Any, Dict, List, Set

from kg_module import KnowledgeGraphStore, normalize_entity
from model_manager import ModelManager


class VerifierAgent:
    """Hybrid verifier: KG query first, then LLM reasoning over relevant subgraph only."""

    def __init__(self, model_manager: ModelManager, kg_store: KnowledgeGraphStore) -> None:
        self.model_manager = model_manager
        self.kg_store = kg_store

    def _extract_key_entities(self, answer: str) -> List[str]:
        candidates = re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{2,}\b", answer)
        stop = {"the", "and", "for", "with", "that", "this", "from", "into", "are", "was", "were"}
        entities = [normalize_entity(c) for c in candidates if c.lower() not in stop]
        uniq: List[str] = []
        seen: Set[str] = set()
        for e in entities:
            if e not in seen:
                seen.add(e)
                uniq.append(e)
        return uniq[:15]

    def _collect_relevant_kg_context(self, entities: List[str]) -> List[Dict[str, Any]]:
        evidence: List[Dict[str, Any]] = []
        for ent in entities:
            evidence.extend(self.kg_store.query_entity(ent))
        unique: List[Dict[str, Any]] = []
        seen = set()
        for rel in evidence:
            key = (rel.get("subject"), rel.get("predicate"), rel.get("object"))
            if key in seen:
                continue
            seen.add(key)
            unique.append(rel)
        return unique

    async def run(self, answer: str) -> Dict[str, Any]:
        entities = self._extract_key_entities(answer)
        relevant_context = self._collect_relevant_kg_context(entities)

        if not relevant_context:
            prompt = (
                "You are a scientific verifier. KG context is empty.\n"
                "Judge answer reliability using model reasoning only.\n"
                "Return STRICT JSON with keys: status, confidence, evidence, feedback, source_of_truth.\n"
                "status must be one of: valid, contradiction, uncertain.\n"
                f"[ANSWER]\n{answer}\n"
            )
            result = await self.model_manager.generate("verifier", prompt)
            return self._parse_or_default(result.get("response", "{}"), default_source="LLM")

        prompt = (
            "You are a scientific contradiction verifier.\n"
            "Compare answer claims against KG facts below and identify contradictions.\n"
            "Use ONLY the provided KG evidence for KG-grounded judgments.\n"
            "Return STRICT JSON with keys: status, confidence, evidence, feedback, source_of_truth.\n"
            "status must be one of: valid, contradiction, uncertain.\n"
            "source_of_truth must be one of: KG, LLM, mixed.\n"
            f"[ANSWER]\n{answer}\n\n"
            f"[KG_RELEVANT_CONTEXT]\n{json.dumps(relevant_context, ensure_ascii=False)}\n"
        )
        result = await self.model_manager.generate("verifier", prompt)
        return self._parse_or_default(
            result.get("response", "{}"),
            default_source="KG",
            default_evidence=relevant_context,
        )

    def _parse_or_default(
        self,
        raw: str,
        default_source: str,
        default_evidence: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {
                "status": "uncertain",
                "confidence": 0.4,
                "evidence": default_evidence or [],
                "feedback": "Verifier did not return valid JSON; please provide clearer factual grounding.",
                "source_of_truth": default_source,
            }

        status = parsed.get("status", "uncertain")
        if status not in {"valid", "contradiction", "uncertain"}:
            status = "uncertain"
        confidence = float(parsed.get("confidence", 0.5))
        evidence = parsed.get("evidence", default_evidence or [])
        if not isinstance(evidence, list):
            evidence = default_evidence or []
        feedback = str(parsed.get("feedback", "No feedback returned by verifier."))
        source_of_truth = parsed.get("source_of_truth", default_source)
        if source_of_truth not in {"KG", "LLM", "mixed"}:
            source_of_truth = default_source

        return {
            "status": status,
            "confidence": max(0.0, min(1.0, confidence)),
            "evidence": evidence,
            "feedback": feedback,
            "source_of_truth": source_of_truth,
        }
