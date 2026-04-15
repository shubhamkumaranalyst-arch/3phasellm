import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


@dataclass
class AgentCallSpan:
    trace_id: str
    iteration: int
    agent_name: str
    started_at: float
    input_payload: Dict[str, Any]


class SessionLogger:
    """Append-only structured logger for multi-agent traces."""

    def __init__(self, log_dir: Path = Path("logs")) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.trace_id = str(uuid.uuid4())
        self.log_path = log_dir / f"session_{timestamp}.json"

    def start_span(self, agent_name: str, iteration: int, input_payload: Dict[str, Any]) -> AgentCallSpan:
        return AgentCallSpan(
            trace_id=self.trace_id,
            iteration=iteration,
            agent_name=agent_name,
            started_at=time.perf_counter(),
            input_payload=input_payload,
        )

    def end_span(self, span: AgentCallSpan, output_payload: Dict[str, Any]) -> None:
        duration = time.perf_counter() - span.started_at
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trace_id": span.trace_id,
            "iteration": span.iteration,
            "agent_name": span.agent_name,
            "input": span.input_payload,
            "output": output_payload,
            "execution_time": duration,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
