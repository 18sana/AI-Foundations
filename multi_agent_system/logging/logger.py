import os
import json
from datetime import datetime
from pathlib import Path

# Absolute path for the log file
LOG_FILE_PATH = Path(__file__).resolve().parent.parent / "logs" / "agent_log.jsonl"

def log_agent_action(agent: str, action: str, reasoning: str):
    """
    Logs an agent action into logs/agent_log.jsonl as a single JSON line.
    """
    # Ensure logs folder exists
    LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent,
        "action": action,
        "reasoning": reasoning
    }

    with open(LOG_FILE_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

def clear_logs():
    """Helper to reset logs during testing."""
    if LOG_FILE_PATH.exists():
        try:
            os.remove(LOG_FILE_PATH)
        except OSError:
            pass
