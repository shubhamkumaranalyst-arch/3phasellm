import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class InterAgentLogger:
    """Simple JSONL logger for inter-agent communication."""

    def __init__(self, log_path: Path = Path("logs/inter_agent.log")) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, payload: Dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "payload": payload,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
