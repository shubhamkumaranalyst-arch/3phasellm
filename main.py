import argparse
import asyncio
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import fitz
import yaml
from ollama import AsyncClient

from logger import InterAgentLogger


DEFAULT_OLLAMA_HOST = "https://ollama-mobius-sales.mobiusdtaas.ai"
DEFAULT_MODELS_PATH = Path("models.yaml")
DEFAULT_VAULT_DIR = Path("/vault")
DEFAULT_KG_PATH = DEFAULT_VAULT_DIR / "knowledge_graph.json"


@dataclass
class UserTurn:
    turn_id: str
    text: str
    file_path: Optional[str] = None
    think: bool = False
    research: bool = False


class Dispatcher:
    """Agent 1 model router with file-type and flag aware routing."""

    def __init__(self, models_path: Path = DEFAULT_MODELS_PATH) -> None:
        self.models_path = models_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        with self.models_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _resolve(self, route_key: str) -> str:
        routes = self.config.get("routing", {})
        registry = self.config.get("registry", {})
        model_alias_or_custom = routes.get(route_key) or routes.get("default")
        return registry.get(model_alias_or_custom, model_alias_or_custom)

    def route(self, turn: UserTurn) -> Dict[str, str]:
        if turn.think:
            return {"route": "think", "model": self._resolve("think")}
        if turn.research:
            return {"route": "research", "model": self._resolve("research")}

        ext = ""
        if turn.file_path:
            ext = Path(turn.file_path).suffix.lower()

        if ext in {".jpg", ".jpeg", ".png"}:
            return {"route": "image", "model": self._resolve("image")}
        if ext in {".mp3", ".wav"}:
            return {"route": "audio", "model": self._resolve("audio_reasoning")}
        if ext in {".pdf", ".txt"}:
            return {"route": "document", "model": self._resolve("document_reasoning")}

        return {"route": "default", "model": self._resolve("default")}


class PerceiverAgent:
    def __init__(self, client: AsyncClient, dispatcher: Dispatcher, logger: InterAgentLogger) -> None:
        self.client = client
        self.dispatcher = dispatcher
        self.logger = logger

    async def _transcribe_audio(self, file_path: str) -> str:
        transcriber_model = self.dispatcher._resolve("audio_transcriber")
        prompt = (
            "You are a local Whisper-style transcription model. "
            f"Transcribe the audio at this local path exactly if possible: {file_path}. "
            "If direct decoding is unavailable, provide best-effort transcription notes."
        )
        response = await self.client.generate(model=transcriber_model, prompt=prompt)
        return response.get("response", "").strip()

    async def _rag_context_for_document(self, file_path: str) -> str:
        path = Path(file_path)
        if path.suffix.lower() == ".txt" and path.exists():
            text = path.read_text(encoding="utf-8", errors="ignore")
            return text[:6000]

        if path.suffix.lower() == ".pdf" and path.exists():
            doc = fitz.open(path)
            try:
                pages: List[str] = []
                for idx, page in enumerate(doc):
                    page_text = page.get_text("text").strip()
                    if page_text:
                        pages.append(f"[Page {idx + 1}]\n{page_text}")
                extracted = "\n\n".join(pages).strip()
                if extracted:
                    return extracted[:16000]
            finally:
                doc.close()

        # For missing/unreadable files, ask model to produce a context scaffold.
        response = await self.client.generate(
            model=self.dispatcher._resolve("document_preprocessor"),
            prompt=(
                "Create a concise context scaffold for this document path. "
                f"Path: {file_path}. Mention that direct parsing may be required if file isn't readable."
            ),
        )
        return response.get("response", "")

    async def run(self, turn: UserTurn, correction_context: Optional[str] = None) -> Dict[str, Any]:
        route = self.dispatcher.route(turn)
        route_name = route["route"]
        model = route["model"]

        user_prompt = turn.text
        message: Dict[str, Any] = {"role": "user", "content": user_prompt}

        if route_name == "image" and turn.file_path:
            message["images"] = [turn.file_path]

        if route_name == "audio" and turn.file_path:
            transcript = await self._transcribe_audio(turn.file_path)
            user_prompt = f"Audio transcript:\n{transcript}\n\nUser request:\n{turn.text}"
            message = {"role": "user", "content": user_prompt}

        if route_name == "document" and turn.file_path:
            context = await self._rag_context_for_document(turn.file_path)
            user_prompt = (
                "Use the following retrieved context to answer accurately.\n\n"
                f"[RAG CONTEXT]\n{context}\n\n"
                f"[USER QUESTION]\n{turn.text}"
            )
            message = {"role": "user", "content": user_prompt}

        if correction_context:
            message["content"] = (
                f"{message['content']}\n\n[SELF-CORRECTION INSTRUCTION]\n{correction_context}\n"
                "Regenerate the answer so it does not contradict known KG facts."
            )

        result = await self.client.chat(model=model, messages=[message])
        content = result.get("message", {}).get("content", "")

        self.logger.log(
            "agent_1_dispatch_complete",
            {
                "turn_id": turn.turn_id,
                "route": route_name,
                "model": model,
                "has_file": bool(turn.file_path),
            },
        )
        return {
            "route": route_name,
            "model": model,
            "response": content,
            "effective_prompt": message["content"],
        }


class GraphLibrarianAgent:
    def __init__(
        self,
        client: AsyncClient,
        logger: InterAgentLogger,
        kg_path: Path = DEFAULT_KG_PATH,
    ) -> None:
        self.client = client
        self.logger = logger
        self.kg_path = kg_path
        self.kg_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_kg_if_missing()

    def _init_kg_if_missing(self) -> None:
        if not self.kg_path.exists():
            self._write_kg(
                {
                    "nodes": [],
                    "edges": [],
                    "metadata": {"last_updated": None, "turn_ids": []},
                }
            )

    def _read_kg(self) -> Dict[str, Any]:
        with self.kg_path.open("r", encoding="utf-8") as f:
            kg = json.load(f)
        kg.setdefault("nodes", [])
        kg.setdefault("edges", [])
        kg.setdefault("metadata", {})
        kg["metadata"].setdefault("last_updated", None)
        kg["metadata"].setdefault("turn_ids", [])
        return kg

    def _write_kg(self, kg: Dict[str, Any]) -> None:
        with self.kg_path.open("w", encoding="utf-8") as f:
            json.dump(kg, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _triple_hash(subject: str, obj: str) -> str:
        digest = hashlib.sha256(f"{subject}::{obj}".encode("utf-8")).hexdigest()
        return digest

    async def _extract_triplets(
        self, user_input: str, model_output: str
    ) -> List[Dict[str, Any]]:
        prompt = (
            "Extract factual triplets from the interaction below.\n"
            "Return STRICT JSON array only with objects using keys: "
            "subject, predicate, object, confidence_score.\n\n"
            f"[INPUT]\n{user_input}\n\n"
            f"[OUTPUT]\n{model_output}\n"
        )
        response = await self.client.generate(model="llama3.3", prompt=prompt)
        raw = response.get("response", "[]")
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

        fallback_triplets: List[Dict[str, Any]] = []
        pattern = r"\b([A-Z][\w.-]+)\s+(is|are|was|were|has|have|uses|supports)\s+([A-Z]?[\w./:-]+)"
        for s, p, o in re.findall(pattern, f"{user_input}\n{model_output}"):
            fallback_triplets.append(
                {
                    "subject": s,
                    "predicate": p,
                    "object": o,
                    "confidence_score": 0.5,
                }
            )
        return fallback_triplets

    async def run(self, turn: UserTurn, agent_output: str) -> Dict[str, Any]:
        kg = self._read_kg()
        if turn.turn_id in set(kg["metadata"].get("turn_ids", [])):
            return {"status": "skipped", "kg": kg, "new_edges": 0}

        triplets = await self._extract_triplets(turn.text, agent_output)
        existing_hashes: Set[str] = {edge.get("so_hash", "") for edge in kg["edges"]}

        now = datetime.now(timezone.utc).isoformat()
        source_file = turn.file_path or "input_stream"
        nodes_by_id = {n["id"]: n for n in kg["nodes"] if "id" in n}

        new_edges = 0
        for tri in triplets:
            s = str(tri.get("subject", "")).strip()
            p = str(tri.get("predicate", "")).strip()
            o = str(tri.get("object", "")).strip()
            confidence = float(tri.get("confidence_score", 0.0))
            if not s or not p or not o:
                continue

            so_hash = self._triple_hash(s, o)
            if so_hash in existing_hashes:
                continue

            for entity in (s, o):
                if entity not in nodes_by_id:
                    nodes_by_id[entity] = {
                        "id": entity,
                        "label": entity,
                        "source_file": source_file,
                        "timestamp": now,
                    }

            kg["edges"].append(
                {
                    "subject": s,
                    "predicate": p,
                    "object": o,
                    "confidence_score": confidence,
                    "so_hash": so_hash,
                    "source_file": source_file,
                    "timestamp": now,
                }
            )
            existing_hashes.add(so_hash)
            new_edges += 1

        kg["nodes"] = sorted(nodes_by_id.values(), key=lambda x: x["id"])
        turns = set(kg["metadata"].get("turn_ids", []))
        turns.add(turn.turn_id)
        kg["metadata"]["turn_ids"] = sorted(turns)
        kg["metadata"]["last_updated"] = now
        self._write_kg(kg)

        self.logger.log(
            "agent_2_graph_update",
            {
                "turn_id": turn.turn_id,
                "new_edges": new_edges,
                "kg_path": str(self.kg_path),
            },
        )
        return {"status": "updated", "kg": kg, "new_edges": new_edges}


class XAIVerifierAgent:
    def __init__(self, client: AsyncClient, dispatcher: Dispatcher, logger: InterAgentLogger) -> None:
        self.client = client
        self.dispatcher = dispatcher
        self.logger = logger

    async def run(self, answer: str, kg: Dict[str, Any]) -> Dict[str, Any]:
        verifier_model = self.dispatcher._resolve("verifier")
        prompt = (
            "You are an expert biomedical XAI verifier.\n"
            "Semantically verify the model answer against the supplied knowledge graph JSON.\n"
            "Flag scientific contradictions, unsupported claims, and confirm supported claims.\n"
            "Return STRICT JSON with keys:\n"
            "trace_lines (array of strings),\n"
            "contradictions (array of objects with claim, reason, conflicting_kg_evidence),\n"
            "requires_self_correction (boolean).\n\n"
            f"[MODEL_ANSWER]\n{answer}\n\n"
            f"[KNOWLEDGE_GRAPH_JSON]\n{json.dumps(kg, ensure_ascii=False)}\n"
        )
        model_result = await self.client.generate(model=verifier_model, prompt=prompt)
        raw_text = model_result.get("response", "{}")

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            parsed = {
                "trace_lines": [
                    "Verifier output was not valid JSON.",
                    "Fallback: treat answer as [Model Inference / Unverified].",
                ],
                "contradictions": [],
                "requires_self_correction": False,
            }

        trace_lines = parsed.get("trace_lines", [])
        contradictions = parsed.get("contradictions", [])
        requires_self_correction = bool(parsed.get("requires_self_correction", bool(contradictions)))
        if not isinstance(trace_lines, list):
            trace_lines = [str(trace_lines)]
        if not isinstance(contradictions, list):
            contradictions = []

        traceability_block = "[TRACEABILITY]\n" + (
            "\n".join(str(line) for line in trace_lines)
            if trace_lines
            else "- No claims evaluated by verifier."
        )
        self.logger.log(
            "agent_3_semantic_verification",
            {
                "verifier_model": verifier_model,
                "contradictions_count": len(contradictions),
            },
        )
        return {
            "traceability_block": traceability_block,
            "contradictions": contradictions,
            "requires_self_correction": requires_self_correction,
        }


class ThreePAKSSystem:
    def __init__(self, ollama_host: str, models_path: Path, kg_path: Path) -> None:
        self.logger = InterAgentLogger()
        self.client = AsyncClient(host=ollama_host)
        self.dispatcher = Dispatcher(models_path=models_path)
        self.perceiver = PerceiverAgent(self.client, self.dispatcher, self.logger)
        self.librarian = GraphLibrarianAgent(self.client, self.logger, kg_path=kg_path)
        self.verifier = XAIVerifierAgent(self.client, self.dispatcher, self.logger)

    async def process_turn(self, turn: UserTurn) -> Dict[str, Any]:
        perceiver_result = await self.perceiver.run(turn)
        librarian_result = await self.librarian.run(turn, perceiver_result["response"])
        verifier_result = await self.verifier.run(perceiver_result["response"], librarian_result["kg"])

        # Hallucination/self-correction loop.
        max_corrections = 1
        attempts = 0
        final_response = perceiver_result["response"]
        final_verifier = verifier_result

        while final_verifier["requires_self_correction"] and attempts < max_corrections:
            attempts += 1
            correction_context = (
                "The verifier found contradictions:\n"
                + json.dumps(final_verifier["contradictions"], indent=2)
            )
            self.logger.log(
                "agent_3_self_correction_trigger",
                {"turn_id": turn.turn_id, "attempt": attempts, "details": final_verifier["contradictions"]},
            )
            retry = await self.perceiver.run(turn, correction_context=correction_context)
            final_response = retry["response"]
            librarian_result = await self.librarian.run(turn, final_response)
            final_verifier = await self.verifier.run(final_response, librarian_result["kg"])

        self.logger.log(
            "turn_complete",
            {
                "turn_id": turn.turn_id,
                "model": perceiver_result["model"],
                "self_correction_attempts": attempts,
            },
        )

        return {
            "turn_id": turn.turn_id,
            "model": perceiver_result["model"],
            "response": final_response,
            "traceability": final_verifier["traceability_block"],
            "contradictions": final_verifier["contradictions"],
            "kg_path": str(self.librarian.kg_path),
        }


def _parse_input_stream(raw: str) -> Tuple[str, Optional[str], bool, bool]:
    tokens = raw.strip().split()
    think = "--think" in tokens
    research = "--research" in tokens
    filtered = [t for t in tokens if t not in {"--think", "--research"}]

    file_path = None
    for tok in filtered:
        if Path(tok).suffix.lower() in {".jpg", ".jpeg", ".png", ".mp3", ".wav", ".pdf", ".txt"}:
            file_path = tok
            break

    text = " ".join(t for t in filtered if t != file_path).strip()
    return text, file_path, think, research


async def interactive_loop(args: argparse.Namespace) -> None:
    system = ThreePAKSSystem(
        ollama_host=args.ollama_host,
        models_path=Path(args.models),
        kg_path=Path(args.kg_path),
    )
    print("3PAKS online. Enter queries; include file paths and --think/--research flags. Type 'exit' to quit.")
    turn_num = 1
    while True:
        raw = input("\nYou> ").strip()
        if raw.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        text, file_path, think, research = _parse_input_stream(raw)
        turn = UserTurn(
            turn_id=f"turn-{turn_num}",
            text=text or "Please summarize.",
            file_path=file_path,
            think=think,
            research=research,
        )
        result = await system.process_turn(turn)
        print("\nAssistant>")
        print(result["response"])
        print("\n" + result["traceability"])
        if result["contradictions"]:
            print("\n[VERIFIER ALERT] Contradictions detected:")
            print(json.dumps(result["contradictions"], indent=2))
        print(f"\n[KG] {result['kg_path']}")
        turn_num += 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="3-Phase Agentic Knowledge System (3PAKS)")
    parser.add_argument("--models", default=str(DEFAULT_MODELS_PATH), help="Path to models.yaml")
    parser.add_argument("--kg-path", default=str(DEFAULT_KG_PATH), help="Path to /vault/knowledge_graph.json")
    parser.add_argument("--ollama-host", default=os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST))
    parser.add_argument("--think", action="store_true", help="Set default think mode for single-shot usage")
    parser.add_argument("--research", action="store_true", help="Set default research mode for single-shot usage")
    parser.add_argument("prompt", nargs="*", help="Optional one-shot input stream")
    return parser


async def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.prompt:
        system = ThreePAKSSystem(
            ollama_host=args.ollama_host,
            models_path=Path(args.models),
            kg_path=Path(args.kg_path),
        )
        raw = " ".join(args.prompt)
        text, file_path, think, research = _parse_input_stream(raw)
        turn = UserTurn(
            turn_id=f"oneshot-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            text=text or "Please summarize.",
            file_path=file_path,
            think=think or args.think,
            research=research or args.research,
        )
        result = await system.process_turn(turn)
        print(result["response"])
        print("\n" + result["traceability"])
        return

    await interactive_loop(args)


if __name__ == "__main__":
    asyncio.run(main())
