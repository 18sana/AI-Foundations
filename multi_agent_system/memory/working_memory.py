from typing import Dict, Any, List, Optional
from multi_agent_system.models.schemas import ExecutionPlan, ReportOutput

class WorkingMemory:
    def __init__(self):
        self.plan: Optional[ExecutionPlan] = None
        self.research_findings: List[str] = []
        self.intermediate_decisions: List[Dict[str, Any]] = []
        self.final_output: Optional[ReportOutput] = None

    def set_plan(self, plan: ExecutionPlan):
        self.plan = plan

    def add_research_finding(self, finding: str):
        self.research_findings.append(finding)

    def log_decision(self, agent: str, decision: str, reason: str):
        self.intermediate_decisions.append({
            "agent": agent,
            "decision": decision,
            "reason": reason
        })

    def set_final_output(self, output: ReportOutput):
        self.final_output = output

    def reset(self):
        self.plan = None
        self.research_findings.clear()
        self.intermediate_decisions.clear()
        self.final_output = None
