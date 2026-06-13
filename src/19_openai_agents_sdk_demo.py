# from dotenv import load_dotenv
# load_dotenv()

# from agents import Agent, Runner

# agent = Agent(
#     name="Math Tutor",
#     instructions=""""
#     You are a helpful math tutor.
#     """
# )

# result = Runner.run_sync(
#     agent,
#     "What is 25 * 12?"
# )

# print(result.final_output)

import requests
from agents import function_tool


@function_tool
def get_weather(
    latitude: float,
    longitude: float
) -> str:

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}"
        f"&longitude={longitude}"
        "&current=temperature_2m"
    )

    data = requests.get(url).json()

    return str(
        data["current"]["temperature_2m"]
    )
agent = Agent(
    name="Weather Agent",
    instructions=""""
    Use weather tool whenever
    weather information is needed.
    """,
    tools=[get_weather]
)
