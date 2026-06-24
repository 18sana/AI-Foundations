from pydantic import BaseModel, Field
from typing import List

class ExecutionPlan(BaseModel):
    goal: str = Field(..., description="The main target or goal of the user request.")
    steps: List[str] = Field(..., description="Chronological steps to accomplish the goal.")

class ReportOutput(BaseModel):
    title: str = Field(..., description="High-level descriptive title of the report.")
    summary: str = Field(..., description="Concise summary summarizing all key research findings.")
    key_points: List[str] = Field(..., description="List of important bullet points detailing the discoveries.")
