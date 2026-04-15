# 3-Phase Agentic Knowledge System (3PAKS)

Async three-agent orchestration using `ollama-python` and a remote Ollama backend.

- Backend default: `https://ollama-mobius-sales.mobiusdtaas.ai`
- Vault default: `/vault/knowledge_graph.json`

## Agents

1. **Agent 1: Multimodal Dispatcher (Perceiver)**
   - Uses `Dispatcher` routing table in `models.yaml`.
   - Routing:
     - `.jpg/.jpeg/.png` -> `llava`
     - `.mp3/.wav` -> local Whisper transcription route, then reason with `llama3.3`
     - `.pdf/.txt` -> RAG context injection, then reason with `llama3.3`
     - `--think` -> `deepseek-r1`
     - `--research` -> `llama3-70b` (or custom model string)

2. **Agent 2: Graph Librarian (Memory)**
   - Extracts `{subject, predicate, object, confidence_score}` triplets from `(input + output)`.
   - Stores graph at `/vault/knowledge_graph.json`.
   - Deduplicates with a hash of `subject + object` and only appends new knowledge.
   - Node schema includes: `id`, `label`, `source_file`, `timestamp`.

3. **Agent 3: XAI Verifier (Auditor)**
   - Cross-references answer claims against KG.
   - Appends a `[TRACEABILITY]` block with source of truth labels.
   - If contradictions are found, triggers a self-correction retry to Agent 1.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Interactive loop:

```bash
python main.py
```

One-shot input stream (with file path + flags in argv):

```bash
python main.py "Summarize sample.txt --research"
python main.py "Explain image.jpg"
python main.py "Transcribe call.wav --think"
```

Override endpoint:

```bash
OLLAMA_HOST="https://ollama-mobius-sales.mobiusdtaas.ai" python main.py
```

## Logs

Inter-agent logs are written to:

- `logs/inter_agent.log`
