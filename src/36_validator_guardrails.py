from pydantic import BaseModel
from pydantic import ValidationError


class SafeAnswer(BaseModel):

    answer: str
    confidence: float


good_output = {
    "answer": "MCP stands for Model Context Protocol.",
    "confidence": 0.95
}

bad_output = {
    "answer": "MCP stands for Model Context Protocol.",
    "confidence": "high"
}

try:

    result = SafeAnswer(
        # **good_output
        **bad_output
    )

    print(
        "\nValidation Passed"
    )

    print(result)

except ValidationError as e:

    print(e)