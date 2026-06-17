import os
from anthropic import Anthropic
from dotenv import load_dotenv
import requests

def get_weather(latitude: float, longitude: float) -> float:
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}"
        f"&longitude={longitude}"
        "&current=temperature_2m"
    )
    response = requests.get(url)
    data = response.json()
    return data["current"]["temperature_2m"]

def run_agent(goal: str) -> str:
    load_dotenv()
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    # If the goal is about weather/temperature, fetch weather and return conversion
    if "weather" in goal.lower() or "temperature" in goal.lower():
        lat, lon = 51.5074, -0.1278  # London coordinates
        temp_c = get_weather(lat, lon)
        temp_f = (temp_c * 9 / 5) + 32
        return f"London temperature is {temp_f:.2f}°F / {temp_c:.2f}°C"
    
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": goal
            }
        ]
    )
    return response.content[0].text
