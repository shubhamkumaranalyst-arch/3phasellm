import argparse
import asyncio
import os
from pathlib import Path
from typing import Optional, Tuple

from ollama import AsyncClient

from agents import LibrarianAgent, PerceiverAgent
from kg_module import KnowledgeGraphStore
from model_manager import ModelManager
from observability import SessionLogger
from orchestrator import SupervisorOrchestrator
from verifier_module import VerifierAgent


DEFAULT_OLLAMA_HOST = "https://ollama-mobius-sales.mobiusdtaas.ai"
DEFAULT_MODELS_PATH = Path("models.yaml")
DEFAULT_GRAPHML_PATH = Path("/vault/knowledge_graph.graphml")
DEFAULT_KG_JSON_PATH = Path("/vault/knowledge_graph.json")


def parse_input_stream(raw: str) -> Tuple[str, Optional[str], bool, bool]:
    tokens = raw.strip().split()
    think = "--think" in tokens
    research = "--research" in tokens
    filtered = [t for t in tokens if t not in {"--think", "--research"}]

    file_path = None
    for token in filtered:
        if Path(token).suffix.lower() in {".jpg", ".jpeg", ".png", ".mp3", ".wav", ".pdf", ".txt"}:
            file_path = token
            break

    query = " ".join(t for t in filtered if t != file_path).strip()
    return query, file_path, think, research


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="3PAKS multi-agent orchestrator")
    parser.add_argument("--ollama-host", default=os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST))
    parser.add_argument("--models", default=str(DEFAULT_MODELS_PATH))
    parser.add_argument("--graphml", default=str(DEFAULT_GRAPHML_PATH))
    parser.add_argument("--kg-json", default=str(DEFAULT_KG_JSON_PATH))
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("prompt", nargs="*")
    return parser


async def build_system(args: argparse.Namespace) -> SupervisorOrchestrator:
    client = AsyncClient(host=args.ollama_host)
    model_manager = ModelManager(client=client, models_path=Path(args.models), request_timeout_seconds=60.0)
    await model_manager.warm_models(
        [
            "default",
            "document_reasoning",
            "think",
            "verifier",
            "librarian_extractor",
        ]
    )

    kg_store = KnowledgeGraphStore(
        graphml_path=Path(args.graphml),
        json_backup_path=Path(args.kg_json),
    )
    perceiver = PerceiverAgent(model_manager=model_manager, kg_store=kg_store)
    librarian = LibrarianAgent(model_manager=model_manager, kg_store=kg_store)
    verifier = VerifierAgent(model_manager=model_manager, kg_store=kg_store)
    session_logger = SessionLogger()
    return SupervisorOrchestrator(
        perceiver=perceiver,
        librarian=librarian,
        verifier=verifier,
        logger=session_logger,
        max_retries=args.max_retries,
    )


async def run_interactive(args: argparse.Namespace) -> None:
    orchestrator = await build_system(args)
    print("3PAKS Orchestrator online. Enter prompt with optional file path and flags. Type 'exit' to quit.")
    while True:
        raw = input("\nYou> ").strip()
        if raw.lower() in {"exit", "quit"}:
            print("Bye.")
            return

        query, file_path, think, research = parse_input_stream(raw)
        result = await orchestrator.run(
            user_query=query or "Please summarize.",
            file_path=file_path,
            think=think,
            research=research,
        )
        state = result["state"]
        print("\nAssistant>")
        print(state["current_answer"])
        print("\nVerification:")
        print(state["verification"])
        print(f"Decision: {result['decision']} | Iteration: {state['iteration']} | Trace ID: {result['trace_id']}")


async def run_oneshot(args: argparse.Namespace) -> None:
    orchestrator = await build_system(args)
    raw = " ".join(args.prompt)
    query, file_path, think, research = parse_input_stream(raw)
    result = await orchestrator.run(
        user_query=query or "Please summarize.",
        file_path=file_path,
        think=think,
        research=research,
    )
    state = result["state"]
    print(state["current_answer"])
    print("\nVerification:")
    print(state["verification"])
    print(f"Decision: {result['decision']} | Iteration: {state['iteration']} | Trace ID: {result['trace_id']}")


async def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.prompt:
        await run_oneshot(args)
        return
    await run_interactive(args)


if __name__ == "__main__":
    asyncio.run(main())
