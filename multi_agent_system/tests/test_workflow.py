import pytest
import os
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from multi_agent_system.models.schemas import ExecutionPlan, ReportOutput
from multi_agent_system.memory.working_memory import WorkingMemory
from multi_agent_system.logging.logger import log_agent_action, clear_logs, LOG_FILE_PATH
from multi_agent_system.tools.knowledge_base import mock_knowledge_base
from multi_agent_system.agents.supervisor import SupervisorAgent
from multi_agent_system.agents.researcher import ResearcherAgent
from multi_agent_system.agents.writer import WriterAgent
from multi_agent_system.main import run_workflow, PERSISTENT_MEMORY_PATH

@pytest.fixture(autouse=True)
def cleanup():
    """Wipe any generated log and persistent memory files before/after each test."""
    clear_logs()
    if PERSISTENT_MEMORY_PATH.exists():
        try:
            os.remove(PERSISTENT_MEMORY_PATH)
        except OSError:
            pass
    yield
    clear_logs()
    if PERSISTENT_MEMORY_PATH.exists():
        try:
            os.remove(PERSISTENT_MEMORY_PATH)
        except OSError:
            pass

def test_pydantic_schemas():
    # Test valid ExecutionPlan
    plan = ExecutionPlan(goal="Test goal", steps=["Step A", "Step B"])
    assert plan.goal == "Test goal"
    assert plan.steps == ["Step A", "Step B"]
    
    # Test valid ReportOutput
    report = ReportOutput(title="Title", summary="Summary text", key_points=["Point 1"])
    assert report.title == "Title"
    assert report.summary == "Summary text"
    assert report.key_points == ["Point 1"]

def test_working_memory():
    mem = WorkingMemory()
    assert mem.plan is None
    assert len(mem.research_findings) == 0
    
    plan = ExecutionPlan(goal="Goal", steps=["Step"])
    mem.set_plan(plan)
    assert mem.plan.goal == "Goal"
    
    mem.add_research_finding("Found detail A")
    assert mem.research_findings == ["Found detail A"]
    
    mem.log_decision("Supervisor", "Decision A", "Reasoning A")
    assert len(mem.intermediate_decisions) == 1
    assert mem.intermediate_decisions[0]["agent"] == "Supervisor"
    
    report = ReportOutput(title="Title", summary="Summary", key_points=["Point"])
    mem.set_final_output(report)
    assert mem.final_output.title == "Title"

def test_logging_system():
    # File should not exist initially due to cleanup
    assert not LOG_FILE_PATH.exists()
    
    log_agent_action("Supervisor", "Start Workflow", "Initiating the test RAG pipeline")
    assert LOG_FILE_PATH.exists()
    
    # Verify JSONL layout
    with open(LOG_FILE_PATH, "r") as f:
        lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["agent"] == "Supervisor"
        assert data["action"] == "Start Workflow"
        assert data["reasoning"] == "Initiating the test RAG pipeline"
        assert "timestamp" in data

def test_knowledge_base_lookup():
    # Standard lookup
    res = mock_knowledge_base("Tell me about model context protocol", simulate_failure=False)
    assert "Topic: MODEL CONTEXT PROTOCOL" in res
    
    # No match
    res_empty = mock_knowledge_base("xyzabc", simulate_failure=False)
    assert "No entries found matching keyword" in res_empty

def test_researcher_tool_failure_resilience():
    researcher = ResearcherAgent()
    mem = WorkingMemory()
    
    # Force mock_knowledge_base tool to raise ConnectionError
    with patch("multi_agent_system.agents.researcher.mock_knowledge_base", side_effect=ConnectionError("Timeout")):
        res = researcher.research("Tell me about MCP", mem)
        
        # Verify it handled exception gracefully without crash and returned warning fallback text
        assert "Fallback state active" in res
        assert len(mem.research_findings) == 1
        assert "Fallback state active" in mem.research_findings[0]
        
        # Confirm the error was logged in agent_log.jsonl
        with open(LOG_FILE_PATH, "r") as f:
            log_data = [json.loads(line) for line in f.readlines()]
            failure_log = [log for log in log_data if log["action"] == "Handle Tool Failure"]
            assert len(failure_log) == 1
            assert "gracefully caught" in failure_log[0]["reasoning"].lower()

def test_supervisor_planning_fallback(monkeypatch):
    supervisor = SupervisorAgent()
    monkeypatch.setattr(supervisor, "client", None)
    mem = WorkingMemory()
    
    # Running without Anthropic Key should fall back locally
    plan, confidence = supervisor.plan_and_delegate("model context protocol", mem)
    assert confidence == 0.95
    assert "MODEL CONTEXT PROTOCOL" in plan.goal.upper()

def test_supervisor_low_confidence_escalation_flow(monkeypatch):
    supervisor = SupervisorAgent()
    monkeypatch.setattr(supervisor, "client", None)
    mem = WorkingMemory()
    
    plan, confidence = supervisor.plan_and_delegate("low confidence request trigger", mem)
    assert confidence == 0.5
    
    # Mock human typing 'y' (approve)
    with patch("builtins.input", return_value="y"):
        approved = supervisor.request_human_approval()
        assert approved is True
        
    # Mock human typing 'n' (deny)
    with patch("builtins.input", return_value="n"):
        approved = supervisor.request_human_approval()
        assert approved is False

def test_writer_structured_report(monkeypatch):
    writer = WriterAgent()
    monkeypatch.setattr(writer, "client", None)
    mem = WorkingMemory()
    mem.add_research_finding("Model Context Protocol is an open standard.")
    
    # Direct validation check using offline fallback format
    report = writer.write_report("mcp", mem)
    assert isinstance(report, ReportOutput)
    assert "mcp" in report.title.lower() or "mock" in report.title.lower() or "technical" in report.title.lower() or "fallback" in report.title.lower()
    assert len(report.key_points) > 0

def test_end_to_end_workflow():
    # Run the full workflow end to end without failures (patch input in case of low-confidence escalation)
    with patch("builtins.input", return_value="y"):
        success = run_workflow("Voyager 1", simulate_failure=False)
        assert success is True
    
    # Confirm facts were written to persistent memory
    assert PERSISTENT_MEMORY_PATH.exists()
    with open(PERSISTENT_MEMORY_PATH, "r") as f:
        data = json.load(f)
        assert len(data) == 1
        assert "Voyager 1" in data[0]["fact"]

def test_researcher_deterministic_tool_failure():
    researcher = ResearcherAgent()
    mem = WorkingMemory()
    res = researcher.research("Tell me about MCP", mem, simulate_failure=False, force_failure=True)
    assert "Fallback state active" in res
    assert len(mem.research_findings) == 1
    assert "Fallback state active" in mem.research_findings[0]

def test_writer_deterministic_validation_retry(monkeypatch):
    writer = WriterAgent()
    monkeypatch.setattr(writer, "client", None)
    mem = WorkingMemory()
    mem.add_research_finding("Model Context Protocol is an open standard.")
    
    report = writer.write_report("mcp", mem, force_validation_failure=True)
    assert isinstance(report, ReportOutput)
    with open(LOG_FILE_PATH, "r") as f:
        lines = f.readlines()
        actions = [json.loads(line)["action"] for line in lines]
        assert "Validation Failure" in actions
        assert "Retry Validation" in actions
