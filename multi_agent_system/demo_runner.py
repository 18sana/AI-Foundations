import os
import sys
import json
import time
from pathlib import Path
from unittest.mock import patch

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from multi_agent_system.main import run_workflow, PERSISTENT_MEMORY_PATH
from multi_agent_system.logging.logger import clear_logs, LOG_FILE_PATH

def print_banner(title: str):
    print("\n" + "="*80)
    print(f" {title.upper()} ".center(80, "="))
    print("="*80)
    time.sleep(1)

def simulated_input(prompt):
    """
    Simulates a human user typing 'y' with realistic delay.
    """
    print(prompt, end="", flush=True)
    time.sleep(1.0) # thinking time
    for char in "y\n":
        print(char, end="", flush=True)
        time.sleep(0.3) # typing speed
    return "y"

def run_demo():
    # 1. Start fresh by clearing old logs/persistent facts
    clear_logs()
    if PERSISTENT_MEMORY_PATH.exists():
        try:
            os.remove(PERSISTENT_MEMORY_PATH)
        except OSError:
            pass

    print_banner("multi-agent system interactive cli demo")
    print("Initializing environment, clearing previous logs & persistent memory...")
    time.sleep(1.5)

    with patch("builtins.input", side_effect=simulated_input):
        # ----------------------------------------------------------------------
        # Scenario 1: Standard Run
        # ----------------------------------------------------------------------
        print_banner("scenario 1: standard multi-agent run")
        print("Query: 'Tell me about the Model Context Protocol (MCP)'")
        print("Expected: Supervisor formulates plan. If confidence is below threshold,")
        print("          escalation will trigger. We will auto-approve it.")
        time.sleep(2)
        run_workflow("Tell me about the Model Context Protocol (MCP)", simulate_failure=False)
        time.sleep(3)

        # ----------------------------------------------------------------------
        # Scenario 2: Low-Confidence Escalation Run
        # ----------------------------------------------------------------------
        print_banner("scenario 2: low-confidence escalation run")
        print("Query: 'escalate and trigger low confidence for unknown topic'")
        print("Expected: Supervisor sets confidence low (< 0.6), pauses execution,")
        print("          prompts for human approval, and resumes after approval.")
        time.sleep(2)
        run_workflow("escalate and trigger low confidence for unknown topic", simulate_failure=False)
        time.sleep(3)

        # ----------------------------------------------------------------------
        # Scenario 3: Researcher Tool Failure Resilience
        # ----------------------------------------------------------------------
        print_banner("scenario 3: researcher tool failure resilience")
        print("Query: 'Tell me about Solid State Battery'")
        print("Expected: Database lookup raises ConnectionError. Researcher catches it,")
        print("          logs error, writes fallback data, and Writer builds valid fallback report.")
        time.sleep(2)
        run_workflow("Tell me about Solid State Battery", simulate_failure=False, force_tool_failure=True)
        time.sleep(3)

        # ----------------------------------------------------------------------
        # Scenario 4: Writer Validation Schema Retry
        # ----------------------------------------------------------------------
        print_banner("scenario 4: writer validation schema retry")
        print("Query: 'Tell me about Voyager 1'")
        print("Expected: Writer's first output validation fails. Writer logs it,")
        print("          triggers retry with feedback override, and succeeds on second attempt.")
        time.sleep(2)
        run_workflow("Tell me about Voyager 1", simulate_failure=False, force_writer_retry=True)
        time.sleep(3)

    # ----------------------------------------------------------------------
    # Verification: Persistent Memory & Logs
    # ----------------------------------------------------------------------
    print_banner("demo summary & verification")
    
    print(f"\n📂 Logs file location: {LOG_FILE_PATH}")
    print("Listing latest 5 entries from logs:")
    print("-" * 80)
    if LOG_FILE_PATH.exists():
        with open(LOG_FILE_PATH, "r") as f:
            lines = f.readlines()
            for line in lines[-10:]:
                print(line.strip())
    else:
        print("Logs not found.")
    print("-" * 80)

    time.sleep(2)

    print(f"\n💾 Persistent Memory entries saved at: {PERSISTENT_MEMORY_PATH}")
    print("-" * 80)
    if PERSISTENT_MEMORY_PATH.exists():
        with open(PERSISTENT_MEMORY_PATH, "r") as f:
            data = json.load(f)
            print(json.dumps(data, indent=2))
    else:
        print("Persistent memory file not found.")
    print("-" * 80)

    print_banner("multi-agent system demo complete")
    print("You can now stop screen recording.")

if __name__ == "__main__":
    run_demo()
