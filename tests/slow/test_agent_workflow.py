# pyrefly: ignore [missing-import]
from src.agent import run_agent


def test_agent_weather_flow():

    result = run_agent(
        "Get London weather"
    )

    assert "temperature" in result.lower()