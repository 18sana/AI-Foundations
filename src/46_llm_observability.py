import csv
import time
from pathlib import Path
from datetime import datetime

# ----------------------------------
# Log File
# ----------------------------------

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "llm_logs.csv"

# ----------------------------------
# Create CSV Header
# ----------------------------------

if not LOG_FILE.exists():

    with open(
        LOG_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "timestamp",
            "query",
            "model",
            "retrieved_docs",
            "tool_used",
            "latency_seconds",
            "response"
        ])

# ----------------------------------
# Logger Function
# ----------------------------------

def log_llm_interaction(
    query,
    model,
    retrieved_docs,
    tool_used,
    latency,
    response
):

    with open(
        LOG_FILE,
        "a",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            datetime.now(),
            query,
            model,
            retrieved_docs,
            tool_used,
            round(latency, 2),
            response
        ])

# ----------------------------------
# Example Usage
# ----------------------------------

if __name__ == "__main__":

    start_time = time.time()

    # Simulated AI Response

    query = "What is MCP?"

    model = "claude-haiku-4-5"

    retrieved_docs = [
        "MCP enables tool access"
    ]

    tool_used = None

    response = (
        "MCP stands for "
        "Model Context Protocol."
    )

    latency = (
        time.time()
        - start_time
    )

    log_llm_interaction(
        query=query,
        model=model,
        retrieved_docs=retrieved_docs,
        tool_used=tool_used,
        latency=latency,
        response=response
    )

    print(
        "\nLogged Successfully"
    )