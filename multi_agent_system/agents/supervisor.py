import os
import json
from typing import Tuple, Optional
from anthropic import Anthropic
from dotenv import load_dotenv

from multi_agent_system.models.schemas import ExecutionPlan
from multi_agent_system.memory.working_memory import WorkingMemory
from multi_agent_system.logging.logger import log_agent_action

class SupervisorAgent:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if self.api_key and self.api_key != "your_actual_anthropic_api_key_here":
            self.client = Anthropic(api_key=self.api_key)
        else:
            self.client = None

    def plan_and_delegate(self, request: str, memory: WorkingMemory) -> Tuple[Optional[ExecutionPlan], float]:
        """
        Generates a structured execution plan and estimates a confidence score.
        Stores the plan inside working memory.
        """
        log_agent_action("Supervisor", "Create Plan", f"Analyzing request: '{request}'")

        if not self.client:
            # Offline/Test Fallback mode
            if any(k in request.lower() for k in ["ambiguous", "escalate", "low confidence", "unknown"]):
                confidence = 0.5
                goal = "Handle low-confidence ambiguous request."
                steps = [
                    "Request human-in-the-loop approval due to ambiguity.",
                    "Log decision logic to shared working memory.",
                    "Terminate or continue based on human signal."
                ]
            else:
                confidence = 0.95
                goal = f"Investigate and report findings for topic: {request}."
                steps = [
                    f"Research topic '{request}' in mock knowledge base.",
                    "Save findings to shared working memory.",
                    "Writer constructs a validated Pydantic report."
                ]
            
            plan = ExecutionPlan(goal=goal, steps=steps)
            memory.set_plan(plan)
            memory.log_decision("Supervisor", "Plan Formulated", f"Goal: {goal} | Confidence: {confidence}")
            return plan, confidence

        # Call live model
        system_prompt = (
            "You are a Supervisor Agent in a multi-agent system. "
            "Analyze the user request, formulate a logical multi-step execution plan, and assess your confidence (0.0 to 1.0).\n\n"
            "If the request contains words like 'ambiguous', 'unknown', 'low confidence', or requests information outside standard "
            "domains (like Model Context Protocol, Solid State Battery, Voyager 1, Quantum Computing), set confidence below 0.6.\n"
            "Format your response as a strict JSON matching this structure:\n"
            "{\n"
            "  \"goal\": \"Main target of the request\",\n"
            "  \"steps\": [\"step 1\", \"step 2\", ...],\n"
            "  \"confidence\": 0.85\n"
            "}"
        )

        try:
            response = self.client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=300,
                system=system_prompt,
                messages=[{"role": "user", "content": request}],
                temperature=0.0
            )
            raw_text = response.content[0].text.strip()
            
            # Robust JSON extraction
            text_clean = raw_text.strip()
            if text_clean.startswith("```"):
                lines = text_clean.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text_clean = "\n".join(lines).strip()
            start = text_clean.find('{')
            end = text_clean.rfind('}')
            if start != -1 and end != -1 and start < end:
                text_clean = text_clean[start:end+1]

            data = json.loads(text_clean)
            
            plan = ExecutionPlan(goal=data["goal"], steps=data["steps"])
            confidence = float(data["confidence"])
            
            memory.set_plan(plan)
            memory.log_decision("Supervisor", "Plan Formulated", f"Goal: {plan.goal} | Confidence: {confidence}")
            return plan, confidence

        except Exception as e:
            # Graceful fallback on API exception
            log_agent_action("Supervisor", "API Error", f"Failed to call API: {str(e)}. Falling back to local plan.")
            if any(k in request.lower() for k in ["ambiguous", "escalate", "low confidence", "unknown"]):
                confidence = 0.5
            else:
                confidence = 0.9
            plan = ExecutionPlan(
                goal=f"Analyze: {request}",
                steps=["Search knowledge base", "Verify output"]
            )
            memory.set_plan(plan)
            return plan, confidence

    def request_human_approval(self) -> bool:
        """
        CLI-based human-in-the-loop escalation prompt.
        """
        log_agent_action("Supervisor", "Request Approval", "Supervisor confidence is low. Triggering human escalation.")
        print("\nSupervisor confidence is low.")
        user_input = input("Do you want to continue? (y/n): ").strip().lower()
        if user_input == "y":
            log_agent_action("Supervisor", "Approval Granted", "Human approved continuing workflow.")
            return True
        else:
            log_agent_action("Supervisor", "Approval Denied", "Human denied approval. Terminating workflow.")
            return False
