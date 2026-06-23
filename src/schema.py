from typing import List
from pydantic import BaseModel, Field

class RAGResponseSchema(BaseModel):
    answer: str = Field(description="The detailed response answering the user query, incorporating citations.")
    citations: List[str] = Field(description="List of exact sources (document filenames or tools) that support the claims.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0 based on available evidence.")
    follow_up_questions: List[str] = Field(description="List of 2-3 logical follow-up questions to continue the topic.")
