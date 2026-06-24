import random

MOCK_DATABASE = {
    "model context protocol": (
        "The Model Context Protocol (MCP) is an open standard that enables developers to build "
        "secure, two-way connections between AI models and their data sources."
    ),
    "solid state battery": (
        "Solid-state batteries use solid electrolytes instead of liquid ones, allowing energy densities "
        "exceeding 500 Wh/kg."
    ),
    "voyager 1": (
        "Voyager 1 is a space probe launched by NASA in 1977, currently traveling in interstellar space "
        "at a distance of over 160 AU from Earth."
    ),
    "quantum computing": (
        "Quantum computing utilizes qubits to perform computations based on superposition and entanglement, "
        "dramatically speeding up certain algorithmic operations."
    )
}

def mock_knowledge_base(query: str, simulate_failure: bool = True, force_failure: bool = False) -> str:
    """
    Simulates database lookup for a search query.
    Occasionally raises an exception to simulate random tool connection errors.
    """
    # 20% random tool failure rate to test Researcher exception resilience, or 100% if force_failure
    if force_failure or (simulate_failure and random.random() < 0.2):
        raise ConnectionError("Database search timed out or connection failed.")

    query_lower = query.lower()
    
    # Simple keyword search
    for topic, content in MOCK_DATABASE.items():
        if topic in query_lower:
            return f"[Knowledge Base Match] Topic: {topic.upper()} | Content: {content}"
            
    return f"[Knowledge Base Match] No entries found matching keyword in '{query}'."
