import requests

def get_weather(latitude, longitude):

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}"
        f"&longitude={longitude}"
        "&current=temperature_2m"
    )

    response = requests.get(url)

    data = response.json()

    return data["current"]["temperature_2m"]
# print(
#     get_weather(
#         51.5074,
#         -0.1278
#     )
# )

import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

tools = [
    {
        "name": "weather",
        "description": "Get current weather using latitude and longitude",
        "input_schema": {
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number"
                },
                "longitude": {
                    "type": "number"
                }
            },
            "required": [
                "latitude",
                "longitude"
            ]
        }
    }
]
if __name__ == "__main__":
    question = input(
        "Ask about weather: "
    )
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=300,
        tools=tools,
        messages=[
            {
                "role": "user",
                "content": question
            }
        ]
    )
    tool_use = response.content[0]

    result = get_weather(
        tool_use.input["latitude"],
        tool_use.input["longitude"]
    )

    print("\nTemperature:")
    print(result)