import os
import sys
import json
import importlib
from pathlib import Path

base_dir = Path(__file__).parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from pydantic import BaseModel, Field
# pyrefly: ignore [missing-import]
from anthropic import Anthropic
from dotenv import load_dotenv

# ---------------------------------------------------------
# Load Environment
# ---------------------------------------------------------
load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

base_dir = Path(__file__).parent.parent
memory_file = base_dir / "data" / "memory.json"
corpus_file = base_dir / "data" / "corpus.txt"

# Ensure directories and files exist
memory_file.parent.mkdir(parents=True, exist_ok=True)
if not memory_file.exists():
    with open(memory_file, "w") as f:
        json.dump([], f)

# ---------------------------------------------------------
# Pydantic Output Schema
# ---------------------------------------------------------
class IntegratedResponse(BaseModel):
    original_query: str = Field(description="The user's original query")
    rewritten_query: str = Field(description="The optimized query for search")
    memory_context: str = Field(description="Information retrieved from memory")
    retrieved_context: list[str] = Field(description="Document snippets retrieved from Chroma")
    tool_used: str | None = Field(description="Name of the tool called, if any")
    tool_argument: str | None = Field(description="Argument passed to the tool, if any")
    tool_result: str | None = Field(description="Result of the tool execution, if any")
    final_answer: str = Field(description="The final answered response to the user")

# Expose existing implementations by importing them directly
calculator_module = importlib.import_module("src.15_calculator_tool")
calculator = calculator_module.calculator

weather_module = importlib.import_module("src.16_weather_tool")
get_weather = weather_module.get_weather

# ---------------------------------------------------------
# Step 1: Query Rewriting
# ---------------------------------------------------------
def rewrite_query(query: str) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=100,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": f"Rewrite the following user query to optimize it for a vector database search. Return ONLY the rewritten query text: '{query}'"
            }
        ]
    )
    return response.content[0].text.strip()

# ---------------------------------------------------------
# Step 2: Memory Lookup
# ---------------------------------------------------------
def lookup_memory(query: str) -> str:
    try:
        with open(memory_file, "r") as f:
            memory = json.load(f)
    except Exception:
        memory = []
        
    if not memory:
        return "No past conversation memory found."
        
    # Search memory entries matching query terms
    matching_entries = []
    for entry in memory:
        q = entry.get("query", "")
        a = entry.get("answer", "")
        # Check if terms overlap
        query_words = set(query.lower().split())
        q_words = set(q.lower().split())
        if query_words & q_words:
            matching_entries.append(f"Past Q: {q}\nPast A: {a}")
            
    if matching_entries:
        return "\n---\n".join(matching_entries)
    return "No relevant past conversation memory matching this query."

# ---------------------------------------------------------
# Step 3: Retrieval (Chroma Search Reuse)
# ---------------------------------------------------------
def chroma_search(query: str) -> list[str]:
    import chromadb
    client_db = chromadb.PersistentClient(path=str(base_dir / "chroma_db"))
    collection = client_db.get_or_create_collection(name="ai_foundations")
    
    # Initialize collection with corpus if empty
    if collection.count() == 0 and corpus_file.exists():
        documents = [
            line.strip()
            for line in corpus_file.read_text().splitlines()
            if line.strip()
        ]
        collection.add(
            ids=[str(i) for i in range(len(documents))],
            documents=documents
        )
        
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    return results["documents"][0] if results["documents"] else []

# ---------------------------------------------------------
# Step 6: Memory Save
# ---------------------------------------------------------
def save_memory(query: str, answer: str):
    try:
        with open(memory_file, "r") as f:
            memory = json.load(f)
    except Exception:
        memory = []
        
    memory.append({
        "query": query,
        "answer": answer
    })
    
    with open(memory_file, "w") as f:
        json.dump(memory, f, indent=2)

# ---------------------------------------------------------
# Pipeline Orchestration
# ---------------------------------------------------------
def run_assistant(query: str) -> IntegratedResponse:
    # 1. Query Rewriting
    rewritten = rewrite_query(query)
    
    # 2. Memory Lookup
    memory_ctx = lookup_memory(query)
    
    # 3. Retrieval
    retrieved_ctx = chroma_search(rewritten)
    context_str = "\n".join(retrieved_ctx)
    
    # 4. Agent Decision
    tools = [
        {
            "name": "calculator",
            "description": "Perform mathematical calculations",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string"
                    }
                },
                "required": ["expression"]
            }
        },
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
                "required": ["latitude", "longitude"]
            }
        }
    ]
    
    system_prompt = f"""You are an integrated assistant. Answer the question using context and tools if needed.
Memory of past conversations:
{memory_ctx}

Chroma retrieved knowledge context:
{context_str}
"""
    
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=300,
        tools=tools,
        messages=[
            {
                "role": "user",
                "content": query
            }
        ],
        system=system_prompt
    )
    
    tool_used = None
    tool_arg = None
    tool_out = None
    final_ans = ""
    
    # Check if a tool was called
    tool_block = None
    for block in response.content:
        if block.type == "tool_use":
            tool_block = block
            break
            
    if tool_block:
        tool_used = tool_block.name
        
        # 5. Tool Execution
        if tool_used == "calculator":
            expr = tool_block.input["expression"]
            tool_arg = expr
            tool_out = calculator(expr)
        elif tool_used == "weather":
            lat = tool_block.input["latitude"]
            lon = tool_block.input["longitude"]
            tool_arg = f"lat: {lat}, lon: {lon}"
            tool_out = str(get_weather(lat, lon))
            
        # Send tool output back to agent to get final answer
        follow_up = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=300,
            tools=tools,
            messages=[
                {
                    "role": "user",
                    "content": query
                },
                {
                    "role": "assistant",
                    "content": response.content
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "content": tool_out
                        }
                    ]
                }
            ],
            system=system_prompt
        )
        final_ans = follow_up.content[0].text
    else:
        final_ans = response.content[0].text
        
    # 6. Memory Save
    save_memory(query, final_ans)
    
    # 7 & 8. Pydantic Validation & Return (Reusing design pattern from 08_structured_output.py)
    from pydantic import ValidationError
    try:
        response_obj = IntegratedResponse(
            original_query=query,
            rewritten_query=rewritten,
            memory_context=memory_ctx,
            retrieved_context=retrieved_ctx,
            tool_used=tool_used,
            tool_argument=tool_arg,
            tool_result=tool_out,
            final_answer=final_ans
        )
        print("\nValidation Successful")
        print("-" * 40)
        return response_obj
    except ValidationError as e:
        print("\nSchema Validation Failed")
        print("-" * 40)
        print(e)
        raise e

if __name__ == "__main__":
    print("\n--- Running Integrated Assistant Demo ---\n")
    
    # Test query 1 (uses vector search)
    print("Testing Vector Retrieval Query...")
    res1 = run_assistant("Tell me about MCP.")
    print(res1.model_dump_json(indent=2))
    print("\n" + "="*50 + "\n")
    
    # Test query 2 (uses calculator)
    print("Testing Calculator Query...")
    res2 = run_assistant("What is 15.5 * 4?")
    print(res2.model_dump_json(indent=2))
    print("\n" + "="*50 + "\n")
    
    # Test query 3 (uses memory)
    print("Testing Memory Context Query...")
    res3 = run_assistant("What was the result of the math problem we just solved?")
    print(res3.model_dump_json(indent=2))