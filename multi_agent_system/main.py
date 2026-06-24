import os
import sys
import json
import time
from pathlib import Path

# Add project root to system path for absolute imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from multi_agent_system.memory.working_memory import WorkingMemory
from multi_agent_system.agents.supervisor import SupervisorAgent
from multi_agent_system.agents.researcher import ResearcherAgent
from multi_agent_system.agents.writer import WriterAgent
from multi_agent_system.logging.logger import log_agent_action, LOG_FILE_PATH

PERSISTENT_MEMORY_PATH = Path(__file__).resolve().parent / "memory" / "persistent_memory.json"

def save_persistent_fact(fact: str):
    """
    Appends a new fact to memory/persistent_memory.json to persist information across runs.
    """
    PERSISTENT_MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    facts = []
    if PERSISTENT_MEMORY_PATH.exists():
        try:
            with open(PERSISTENT_MEMORY_PATH, "r") as f:
                facts = json.load(f)
                if not isinstance(facts, list):
                    facts = []
        except Exception:
            facts = []
            
    facts.append({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fact": fact
    })
    
    with open(PERSISTENT_MEMORY_PATH, "w") as f:
        json.dump(facts, f, indent=2)
    print(f"-> Fact saved to persistent memory: '{fact}'")

def run_workflow(request: str, simulate_failure: bool = True, force_tool_failure: bool = False, force_writer_retry: bool = False) -> bool:
    """
    Executes the multi-agent RAG workflow for a given request.
    """
    print("\n" + "="*60)
    print(f"STARTING WORKFLOW FOR REQUEST: '{request}'")
    print("="*60)

    # Initialize memory and agents
    memory = WorkingMemory()
    supervisor = SupervisorAgent()
    researcher = ResearcherAgent()
    writer = WriterAgent()

    # Step 1: Supervisor Planning
    print("\n[Step 1] Supervisor Planning...")
    plan, confidence = supervisor.plan_and_delegate(request, memory)
    print(f"Goal: {plan.goal}")
    print(f"Steps: {plan.steps}")
    print(f"Confidence Score: {confidence:.2f}")

    # Step 2: Human Escalation Check
    if confidence < 0.6:
        print("\n[Escalation] Supervisor confidence is below threshold (< 0.6)!")
        approved = supervisor.request_human_approval()
        if not approved:
            print("\nExecution terminated by human. Safe exit.")
            return False
        print("\nResuming execution after human approval...")
    else:
        print("\nSupervisor confidence is sufficient. Continuing without escalation.")

    # Step 3: Researcher Research
    print("\n[Step 2] Researcher Retrieving Data...")
    # Generate search query based on request
    findings = researcher.research(request, memory, simulate_failure=simulate_failure, force_failure=force_tool_failure)
    print(f"Researcher output successfully written to Working Memory.")
    print(f"Findings excerpt: {findings[:150]}...")

    # Step 4: Writer Output Generation & Validation
    print("\n[Step 3] Writer Formatting Report...")
    report = writer.write_report(request, memory, force_validation_failure=force_writer_retry)
    
    if report:
        print("\n" + "-"*40)
        print("FINAL STRUCTURED REPORT (Validated by Pydantic)")
        print("-"*40)
        print(f"Title:   {report.title}")
        print(f"Summary: {report.summary}")
        print(f"Key Points:")
        for pt in report.key_points:
            print(f"  - {pt}")
        print("-"*40)
        
        # Save to persistent memory
        log_agent_action("System", "Save Persistent Memory", f"Saving title '{report.title}' to persistent memory.")
        save_persistent_fact(f"{report.title}: {report.summary}")
    else:
        print("\nWriter failed to compile a valid report.")

    print("\nWorkflow completed successfully.")
    return True

def cli_menu():
    """
    Interactive CLI menu showcasing the required demo scenarios.
    """
    while True:
        print("\n" + "="*45)
        print("  MULTI-AGENT SYSTEM DEMO CLI MENU")
        print("="*45)
        print("1. Scenario 1: Standard Run (Model Context Protocol)")
        print("2. Scenario 2: Escalation Run (Low confidence / ambiguous query)")
        print("3. Scenario 3: Researcher Tool Failure Run (Guarantees database timeout/failure)")
        print("4. Scenario 4: Writer Validation Retry Run (Simulates schema validation failure)")
        print("5. Enter Custom Request")
        print("6. View Logs Path & Persistent Memory")
        print("7. Exit")
        print("="*45)
        choice = input("Select an option (1-7): ").strip()

        if choice == "1":
            run_workflow("Tell me about the Model Context Protocol (MCP)", simulate_failure=False)
        elif choice == "2":
            run_workflow("escalate and trigger low confidence for unknown topic", simulate_failure=False)
        elif choice == "3":
            run_workflow("Tell me about Solid State Battery", simulate_failure=False, force_tool_failure=True)
        elif choice == "4":
            run_workflow("Tell me about Voyager 1", simulate_failure=False, force_writer_retry=True)
        elif choice == "5":
            req = input("\nEnter your custom research query: ").strip()
            if req:
                run_workflow(req, simulate_failure=True)
        elif choice == "6":
            print(f"\n* Logs stored at: {LOG_FILE_PATH}")
            if PERSISTENT_MEMORY_PATH.exists():
                print(f"* Persistent Memory entries:")
                with open(PERSISTENT_MEMORY_PATH, "r") as f:
                    try:
                        mem_data = json.load(f)
                        print(json.dumps(mem_data, indent=2))
                    except Exception:
                        print(f.read())
            else:
                print("* No persistent memory file found yet (run a workflow first).")
        elif choice == "7":
            print("\nExiting. Thank you!")
            break
        else:
            print("\nInvalid choice. Please choose between 1 and 7.")

if __name__ == "__main__":
    cli_menu()
