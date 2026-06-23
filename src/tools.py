import ast
import operator
import random
from typing import Dict, Any, List

# Safe AST Calculator
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos
}

def _safe_eval(node):
    if isinstance(node, ast.Num):  # python < 3.8
        return node.n
    elif isinstance(node, ast.Constant):  # python >= 3.8
        return node.value
    elif isinstance(node, ast.BinOp):
        op = type(node.op)
        if op in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[op](_safe_eval(node.left), _safe_eval(node.right))
    elif isinstance(node, ast.UnaryOp):
        op = type(node.op)
        if op in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[op](_safe_eval(node.operand))
    raise ValueError(f"Unsupported math operator: {node}")

def calculator(expression: str) -> str:
    """
    Safely calculates simple mathematical expressions using AST evaluation.
    Example: calculator("2 * (3 + 4)") -> "14"
    """
    try:
        # Strip potential wrapping characters
        clean_expr = expression.strip().replace(" ", "")
        tree = ast.parse(clean_expr, mode='eval')
        res = _safe_eval(tree.body)
        return str(res)
    except Exception as e:
        return f"Error: Invalid or unsafe expression: {str(e)}"

# Mock Web Search Index
MOCK_WEB_PAGES = {
    "model context protocol mcp": (
        "The Model Context Protocol (MCP) is an open standard developed by Anthropic that enables developers to build "
        "secure, two-way connections between AI models and their data sources. MCP defines a protocol client-server architecture "
        "allowing models to discover tools, search files, read databases, and trigger actions. Key components include MCP clients "
        "(like Claude Desktop or IDE extensions) and MCP servers that expose resources and tools. It establishes a unified "
        "standard, reducing the need to write custom integration code for every API."
    ),
    "solid state battery energy density limit": (
        "While conventional lithium-ion batteries are capped around 350 Wh/kg, solid-state batteries are anticipated to exceed "
        "500 Wh/kg. Companies like QuantumScape and Solid Power are testing solid-state cells with silicon or lithium metal "
        "anodes that boast theoretical limits near 800 Wh/kg. The solid electrolyte prevents dendrite formation, significantly "
        "lowering thermal runaway risk."
    ),
    "voyager 1 current distance from earth": (
        "As of 2026, Voyager 1 is approximately 163 Astronomical Units (AU) or 24.4 billion kilometers from Earth. It remains the "
        "farthest human-made object in space, traveling at a speed of roughly 38,000 miles per hour relative to the Sun. At this "
        "distance, a radio signal takes over 22.5 hours to travel one way between Earth and the spacecraft."
    )
}

def web_search(query: str) -> str:
    """
    Simulates a web search engine. Returns top hits from a mock database.
    Has a 20% probability of returning empty results as per constraints.
    """
    # 20% empty response rate to test non-hallucination guardrails
    if random.random() < 0.20:
        print("Web search simulated network timeout / empty results (20% boundary).")
        return "Search returned 0 results."
        
    query_lower = query.lower()
    
    # Try exact or partial matches
    best_match = None
    for key, content in MOCK_WEB_PAGES.items():
        # Check if all words in key exist in query or vice versa
        if any(word in query_lower for word in key.split()):
            best_match = content
            break
            
    if best_match:
        return f"[Web Search Result] Query: '{query}' -> Content: {best_match}"
    else:
        return f"Search returned 0 results for '{query}'."

# Mock URL summary database
MOCK_URLS = {
    "https://wikipedia.org/wiki/Model_Context_Protocol": (
        "The Model Context Protocol (MCP) page details the architectural design of MCP, a standard proposed in 2024. "
        "It explains that MCP is designed to expose local tools and files to Large Language Models in a standardized manner. "
        "By defining JSON-RPC 2.0 primitives, servers can expose 'prompts', 'resources', and 'tools'. MCP handles context "
        "negotiation, protocol versions, and transport mechanisms like Standard I/O (stdio) or Server-Sent Events (SSE)."
    ),
    "https://nasa.gov/artemis": (
        "Artemis mission dashboard. NASA's program to establish a sustainable human presence on the Moon. "
        "Includes details on SLS Rocket, Orion Spacecraft, and Gateway Station. Artemis II is scheduled for crewed lunar flyby, "
        "and Artemis III will land the first woman and next man at the lunar South Pole. It also details collaborations with commercial "
        "partners like SpaceX and Blue Origin for Human Landing Systems (HLS)."
    )
}

def summarise_doc(url: str) -> str:
    """
    Summarizes content from a given URL. Truncates outputs to 500 words/tokens.
    """
    cleaned_url = url.strip().strip("'\"")
    if cleaned_url in MOCK_URLS:
        text = MOCK_URLS[cleaned_url]
    else:
        text = f"Document content fetched from URL: {url}. It contains various technical specifications and overview summaries."
        
    # Mock token truncation (limit to 500 words)
    words = text.split()
    if len(words) > 500:
        words = words[:500]
        text = " ".join(words) + " [Truncated...]"
        
    return f"[Document Summary: {url}] -> {text}"
