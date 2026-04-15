# 3PAKS: 3-Phase Agentic Knowledge System

This repo runs a true cyclic multi-agent orchestrator on Ollama.

## Core architecture

- **SupervisorOrchestrator**: cyclic control loop with shared mutable `STATE`.
- **PerceiverAgent**: multimodal dispatcher + answer generation.
- **LibrarianAgent**: strict triplet extraction + graph persistence.
- **VerifierAgent**: hybrid KG-query + LLM semantic contradiction check.
- **KnowledgeGraphStore**: NetworkX `nx.DiGraph` persisted as GraphML + JSON backup.
- **SessionLogger**: append-only structured observability logs.
- **ModelManager**: lazy model registry resolution, warming, timeout + fallback.

## Shared STATE

```python
{
  "user_query": str,
  "current_answer": str,
  "knowledge_context": list,
  "verification": dict,
  "iteration": int
}
```

## Verification output

```python
{
  "status": "valid" | "contradiction" | "uncertain",
  "confidence": float,
  "evidence": list,
  "feedback": str,
  "source_of_truth": "KG" | "LLM" | "mixed"
}
```

## Storage

- Graph primary: `/vault/knowledge_graph.graphml`
- JSON backup: `/vault/knowledge_graph.json`
- Session logs: `./logs/session_<timestamp>.json`

## Run

```bash
pip install -r requirements.txt
python main.py
```

One-shot examples:

```bash
python main.py "Summarize study.pdf --research"
python main.py "Explain chest_xray.png"
python main.py "Transcribe consult.wav --think"
```

## Model registry check

`models.yaml` maps think/verifier to `deepseek-r1:32b`.
Confirm this tag exists on your Mobius endpoint (`https://ollama-mobius-sales.mobiusdtaas.ai`).
If your deployed tag differs, update `models.yaml`.
