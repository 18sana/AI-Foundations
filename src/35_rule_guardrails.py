import re

BLOCKED_TOPICS = [
    "password",
    "credit card",
    "hack",
    "ssn"
]

def check_input(query: str):

    for topic in BLOCKED_TOPICS:

        if topic.lower() in query.lower():

            return False

    return True


while True:

    query = input("\nQuery: ")

    if not check_input(query):

        print(
            "\nBlocked by Guardrail"
        )

        continue

    print(
        "\nAllowed:",
        query
    )