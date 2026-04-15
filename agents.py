from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz

from kg_module import KnowledgeGraphStore, normalize_entity
from model_manager import ModelManager


LIBRARIAN_EXTRACTION_PROMPT = """You are a knowledge graph extraction engine.
Task: extract only factual, reusable, domain-stable knowledge from input/output text.

Hard rules:
1) Output STRICT JSON only as an array of objects.
2) Each object must contain exactly: subject, predicate, object, confidence.
3) Normalize entities to lowercase singular form.
4) Ignore conversational, temporary, subjective, or procedural statements.
5) Do not include duplicate triples.
6) Do NOT use vague predicates like "related_to" or "associated_with".
7) Prefer ontology-style predicates from this set when possible:
   - is_a
   - part_of
   - causes
   - uses
   - located_in

Few-shot examples:
Example 1
Input: "Lungs are part of the respiratory systems."
Output:
[
  {"subject":"lung","predicate":"part_of","object":"respiratory system","confidence":0.97}
]

Example 2
Input: "Ibuprofen is a nonsteroidal anti-inflammatory drug used for pain relief."
Output:
[
  {"subject":"ibuprofen","predicate":"is_a","object":"nonsteroidal anti-inflammatory drug","confidence":0.98},
  {"subject":"ibuprofen","predicate":"uses","object":"pain relief","confidence":0.86}
]

Example 3
Input: "Smoking causes lung cancer, and John said he might quit tomorrow."
Output:
[
  {"subject":"smoking","predicate":"causes","object":"lung cancer","confidence":0.99}
]

Now process the following text and return JSON only.
"""


class PerceiverAgent:
    def __init__(self, model_manager: ModelManager) -> None:
        self.model_manager = model_manager

    async def _transcribe_audio(self, file_path: str) -> str:
        prompt = (
            "Transcribe this local audio path faithfully and output plain transcript text. "
            f"Audio path: {file_path}"
        )
        result = await self.model_manager.generate("audio_transcriber", prompt)
        return result.get("response", "").strip()

    async def _parse_document(self, file_path: str) -> str:
        path = Path(file_path)
        if path.suffix.lower() == ".txt" and path.exists():
            return path.read_text(encoding="utf-8", errors="ignore")[:20000]

        if path.suffix.lower() == ".pdf" and path.exists():
            doc = fitz.open(path)
            try:
                pages = []
                for idx, page in enumerate(doc):
                    text = page.get_text("text").strip()
                    if text:
                        pages.append(f"[Page {idx + 1}]\n{text}")
                return "\n\n".join(pages)[:30000]
            finally:
                doc.close()

        return ""

    async def run(
        self,
        user_query: str,
        file_path: Optional[str] = None,
        think: bool = False,
        research: bool = False,
        feedback: Optional[str] = None,
    ) -> Dict[str, Any]:
        route_key = "default"
        message: Dict[str, Any] = {"role": "user", "content": user_query}

        if think:
            route_key = "think"
        elif research:
            route_key = "research"

        if file_path:
            suffix = Path(file_path).suffix.lower()
            if suffix in {".jpg", ".jpeg", ".png"}:
                route_key = "image"
                message["images"] = [file_path]
            elif suffix in {".mp3", ".wav"}:
                route_key = "audio_reasoning"
                transcript = await self._transcribe_audio(file_path)
                message["content"] = f"Audio transcript:\n{transcript}\n\nQuestion:\n{user_query}"
            elif suffix in {".pdf", ".txt"}:
                route_key = "document_reasoning"
                doc_text = await self._parse_document(file_path)
                message["content"] = (
                    "Use the following clinical document context to answer with evidence.\n\n"
                    f"[DOCUMENT_CONTEXT]\n{doc_text}\n\n"
                    f"[QUESTION]\n{user_query}"
                )

        if feedback:
            message["content"] = (
                f"{message['content']}\n\n[VERIFIER FEEDBACK]\n{feedback}\n"
                "Revise your answer to eliminate contradictions and improve factual consistency."
            )

        result = await self.model_manager.chat(route_key, [message])
        return {
            "answer": result.get("message", {}).get("content", ""),
            "route_key": route_key,
        }


class LibrarianAgent:
    def __init__(self, model_manager: ModelManager, kg_store: KnowledgeGraphStore) -> None:
        self.model_manager = model_manager
        self.kg_store = kg_store

    async def extract_triples(self, user_query: str, answer: str) -> List[Dict[str, Any]]:
        prompt = (
            f"{LIBRARIAN_EXTRACTION_PROMPT}\n"
            f"[USER_QUERY]\n{user_query}\n\n"
            f"[MODEL_ANSWER]\n{answer}\n"
        )
        result = await self.model_manager.generate("librarian_extractor", prompt)
        raw = result.get("response", "[]")
        try:
            triples = __import__("json").loads(raw)
            if isinstance(triples, list):
                normalized: List[Dict[str, Any]] = []
                seen = set()
                for t in triples:
                    subject = normalize_entity(str(t.get("subject", "")))
                    predicate = str(t.get("predicate", "")).strip().lower()
                    obj = normalize_entity(str(t.get("object", "")))
                    confidence = float(t.get("confidence", 0.0))
                    key = (subject, predicate, obj)
                    if not subject or not predicate or not obj or key in seen:
                        continue
                    seen.add(key)
                    normalized.append(
                        {
                            "subject": subject,
                            "predicate": predicate,
                            "object": obj,
                            "confidence": max(0.0, min(1.0, confidence)),
                        }
                    )
                return normalized
        except Exception:
            pass
        return []

    async def run(self, user_query: str, answer: str, source_file: Optional[str]) -> Dict[str, Any]:
        triples = await self.extract_triples(user_query, answer)
        added = self.kg_store.add_triples(triples, source_file=source_file)
        return {
            "triples": triples,
            "added_count": added,
        }
