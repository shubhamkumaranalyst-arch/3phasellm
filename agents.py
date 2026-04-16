import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz

from kg_module import KnowledgeGraphStore, normalize_entity
from model_manager import ModelManager


LIBRARIAN_EXTRACTION_PROMPT = """You are a Knowledge Graph extraction engine.

Your task is to extract ONLY factual, reusable knowledge as triples.

STRICT RULES:

* Output ONLY valid JSON (no explanation, no text outside JSON)
* Do NOT hallucinate
* Extract ONLY facts explicitly present
* Ignore opinions, questions, or conversational text
* Normalize all entities:

  * lowercase
  * singular form
  * no punctuation
* Use ONLY these predicates:

  * is_a
  * part_of
  * uses
  * located_in
  * causes
  * depends_on
  * derived_from

FORMAT:
[
{
"subject": "string",
"predicate": "string",
"object": "string",
"confidence": float (0.0 to 1.0)
}
]

CONSTRAINTS:

* No duplicate triples
* No vague predicates like 'related_to'
* Each triple must stand alone as a fact

FEW-SHOT EXAMPLES:

Input:
"Llama3 is a large language model developed by Meta."

Output:
[
{
"subject": "llama3",
"predicate": "is_a",
"object": "large language model",
"confidence": 0.95
},
{
"subject": "llama3",
"predicate": "derived_from",
"object": "meta",
"confidence": 0.9
}
]

Input:
"Whisper is used for speech recognition."

Output:
[
{
"subject": "whisper",
"predicate": "uses",
"object": "speech recognition",
"confidence": 0.9
}
]

Now extract triples from the given input."""


class PerceiverAgent:
    def __init__(self, model_manager: ModelManager, kg_store: KnowledgeGraphStore) -> None:
        self.model_manager = model_manager
        self.kg_store = kg_store

    def _extract_query_entities(self, query: str) -> List[str]:
        tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{2,}\b", query)
        stop = {"the", "and", "for", "with", "that", "this", "from", "into", "are", "was", "were"}
        entities: List[str] = []
        seen = set()
        for token in tokens:
            if token.lower() in stop:
                continue
            norm = normalize_entity(token)
            if norm and norm not in seen:
                seen.add(norm)
                entities.append(norm)
        return entities[:10]

    def _get_kg_context(self, query: str, max_items: int = 10) -> List[Dict[str, Any]]:
        entities = self._extract_query_entities(query)
        context: List[Dict[str, Any]] = []
        seen = set()
        for entity in entities:
            rels = self.kg_store.query_entity(entity)
            for rel in rels:
                key = (rel.get("subject"), rel.get("predicate"), rel.get("object"))
                if key in seen:
                    continue
                seen.add(key)
                context.append(rel)
                if len(context) >= max_items:
                    return context
        return context

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

        if think:
            route_key = "think"
        elif research:
            route_key = "research"

        kg_context = self._get_kg_context(user_query, max_items=10)
        prompt = (
            "You are a reasoning agent.\n\n"
            f"Context from Knowledge Graph:\n{json.dumps(kg_context, ensure_ascii=False)}\n\n"
            f"User query:\n{user_query}\n\n"
            "Instructions:\n\n"
            "* Use KG context as primary source of truth\n"
            "* If insufficient, use general reasoning\n"
            "* Do NOT contradict KG\n\n"
            "Generate answer."
        )

        message: Dict[str, Any] = {"role": "user", "content": prompt}

        if file_path:
            suffix = Path(file_path).suffix.lower()
            if suffix in {".jpg", ".jpeg", ".png"}:
                route_key = "image"
                message["images"] = [file_path]
            elif suffix in {".mp3", ".wav"}:
                route_key = "audio_reasoning"
                transcript = await self._transcribe_audio(file_path)
                message["content"] = (
                    f"{prompt}\n\n[TRANSCRIPT]\n{transcript}\n"
                )
            elif suffix in {".pdf", ".txt"}:
                route_key = "document_reasoning"
                doc_text = await self._parse_document(file_path)
                message["content"] = (
                    f"{prompt}\n\n[DOCUMENT_CONTEXT]\n{doc_text}\n"
                )

        if feedback:
            message["content"] = (
                f"{message['content']}\n\nPrevious answer was incorrect because: {feedback}. Improve it."
            )

        result = await self.model_manager.chat(route_key, [message])
        return {
            "answer": result.get("message", {}).get("content", ""),
            "kg_context": kg_context,
        }


class LibrarianAgent:
    def __init__(self, model_manager: ModelManager, kg_store: KnowledgeGraphStore) -> None:
        self.model_manager = model_manager
        self.kg_store = kg_store

    async def extract_triples(self, user_query: str, answer: str) -> List[Dict[str, Any]]:
        prompt = (
            f"{LIBRARIAN_EXTRACTION_PROMPT}\n\n"
            f"Input:\n{user_query}\n\n"
            f"Input:\n{answer}\n"
        )
        result = await self.model_manager.generate("librarian_extractor", prompt)
        raw = result.get("response", "[]")
        try:
            triples = json.loads(raw)
            if isinstance(triples, list):
                normalized: List[Dict[str, Any]] = []
                seen = set()
                allowed_predicates = {
                    "is_a",
                    "part_of",
                    "uses",
                    "located_in",
                    "causes",
                    "depends_on",
                    "derived_from",
                }
                for t in triples:
                    subject = normalize_entity(str(t.get("subject", "")))
                    predicate = str(t.get("predicate", "")).strip().lower()
                    obj = normalize_entity(str(t.get("object", "")))
                    confidence = float(t.get("confidence", 0.0))
                    key = (subject, predicate, obj)
                    if (
                        not subject
                        or not predicate
                        or not obj
                        or predicate not in allowed_predicates
                        or key in seen
                    ):
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
