# 3PAKS: 3-Phase Agentic Knowledge System

This repo runs a strict cyclic multi-agent orchestrator on Ollama.

## Core architecture

- **SupervisorOrchestrator**: cyclic control loop with shared mutable `STATE`.
- **PerceiverAgent**: KG-first reasoning prompt + multimodal handling.
- **LibrarianAgent**: strict ontology-driven triple extraction.
- **VerifierAgent**: strict hybrid KG + LLM verification.
- **KnowledgeGraphStore**: NetworkX `nx.DiGraph` with GraphML + JSON persistence.
- **SessionLogger**: append-only structured observability logs.
- **ModelManager**: lazy model routing, warm-up, timeout, fallback.

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

## Decision policy

- ACCEPT only when `status == "valid"` and `confidence >= 0.6`.
- Otherwise retry with verifier feedback injected into Perceiver.
- Max retries = 3.
- Retry exhaustion returns best attempt with warning: `Low confidence answer`.

## Knowledge Graph

- Graph primary: `/vault/knowledge_graph.graphml`
- JSON backup: `/vault/knowledge_graph.json`
- Canonical map: `canonical_map.json`
- Entity normalization is applied on store and query.

## Run

```bash
pip install -r requirements.txt
python main.py
```

## Model registry check

`models.yaml` maps think/verifier to `deepseek-r1:32b`.
Confirm this model tag exists on your Mobius endpoint (`https://ollama-mobius-sales.mobiusdtaas.ai`).
