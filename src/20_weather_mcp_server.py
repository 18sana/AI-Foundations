from mcp.server.fastmcp import FastMCP
import requests

mcp = FastMCP("Weather Server")


@mcp.tool()
def get_weather(
    latitude: float,
    longitude: float
) -> str:
    """
    Get current temperature.
    """

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}"
        f"&longitude={longitude}"
        "&current=temperature_2m"
    )

    data = requests.get(url).json()

    temp = data["current"]["temperature_2m"]

    return f"{temp}°C"


if __name__ == "__main__":
    mcp.run()