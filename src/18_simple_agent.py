# import os

# from anthropic import Anthropic
# from dotenv import load_dotenv

# load_dotenv()

# client = Anthropic(
#     api_key=os.getenv("ANTHROPIC_API_KEY")
# )
# def calculator(expression):
#     try:
#         return str(eval(expression))
#     except Exception as e:
#         return f"Error: {e}"

# tools = [
#     {
#         "name": "calculator",
#         "description": "Perform mathematical calculations",
#         "input_schema": {
#             "type": "object",
#             "properties": {
#                 "expression": {
#                     "type": "string"
#                 }
#             },
#             "required": ["expression"]
#         }
#     }
# ]
# goal = input(
#     "\nEnter Goal: "
# )
# MAX_STEPS = 5

# messages = [
#     {
#         "role": "user",
#         "content": goal
#     }
# ]
# for step in range(MAX_STEPS):

#     print(f"\n--- Step {step + 1} ---")

#     response = client.messages.create(
#         model="claude-haiku-4-5",
#         max_tokens=300,
#         tools=tools,
#         messages=messages
#     )
#     tool_use = None

#     for block in response.content:
#         if block.type == "tool_use":
#             tool_use = block

#     if not tool_use:
#         print("\nFinal Answer:\n")
#         print(response.content[0].text)
#         break

#     print(
#         f"Using Tool: {tool_use.name}"
#     )

#     result = calculator(
#         tool_use.input["expression"]
#     )

#     print(
#         f"Observation: {result}"
#     )
#     messages.append(
#         {
#             "role": "assistant",
#             "content": response.content
#         }
#     )

#     messages.append(
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "tool_result",
#                     "tool_use_id": tool_use.id,
#                     "content": result
#                 }
#             ]
#         }
#     )


import requests


# Tool
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


# Agent Loop
MAX_STEPS = 3

goal = "Get London's temperature and convert it to Fahrenheit"

observation = None

for step in range(MAX_STEPS):

    print(f"\n--- Step {step + 1} ---")

    # Step 1: Retrieve weather
    if step == 0:

        print("Action: Get Weather")

        temperature_c = get_weather(
            51.5074,   # London
            -0.1278
        )

        observation = temperature_c

        print(
            f"Observation: {temperature_c}°C"
        )

    # Step 2: Convert
    elif step == 1:

        print(
            "Action: Convert to Fahrenheit"
        )

        temperature_f = (
            observation * 9 / 5
        ) + 32

        observation = temperature_f

        print(
            f"Observation: {temperature_f:.2f}°F"
        )

    # Step 3: Final Answer
    else:

        print("\nFinal Answer:")

        print(
            f"London temperature is "
            f"{temperature_f:.2f}°F"
        )

        break