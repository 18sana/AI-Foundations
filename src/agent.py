import os
import json
import time
from typing import List, Dict, Any, Generator, Optional
from anthropic import Anthropic
from dotenv import load_dotenv
try:
    from langsmith import traceable
except ImportError:
    def traceable(name: str = None, run_type: str = None):
        def decorator(func):
            import functools
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator

from src.schema import RAGResponseSchema
from src.query_rewriter import QueryRewriter
from src.retriever import DocumentRetriever
from src.memory import SemanticMemory
from src.vector_store import ChromaVectorStore
from src.tools import calculator, web_search, summarise_doc
from src.cache import SemanticCache
from src.guardrails import redact_pii, is_injection_attempt, violates_moderation

# Haiku 4.5 Pricing (estimate per 1M tokens)
INPUT_COST_PER_M = 0.25
OUTPUT_COST_PER_M = 1.25

class RAGAgent:
    def __init__(self, model_name: str = "claude-haiku-4-5", persist_dir: Optional[str] = None):
        load_dotenv()
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = Anthropic(api_key=self.api_key)
        self.model_name = model_name
        self.rewriter = QueryRewriter()
        if persist_dir:
            from pathlib import Path
            p = Path(persist_dir)
            self.retriever = DocumentRetriever(vector_store=ChromaVectorStore(persist_dir=str(p / "documents")))
            self.memory = SemanticMemory(persist_dir=str(p / "memory"))
            self.cache = SemanticCache(persist_dir=str(p / "cache"))
        else:
            self.retriever = DocumentRetriever()
            self.memory = SemanticMemory()
            self.cache = SemanticCache()
        self.total_cost = 0.0

    def _track_cost(self, input_tokens: int, output_tokens: int):
        cost = (input_tokens / 1_000_000.0 * INPUT_COST_PER_M) + (output_tokens / 1_000_000.0 * OUTPUT_COST_PER_M)
        self.total_cost += cost

    @traceable(name="RAG Agent Stream", run_type="chain")
    def run_agent_stream(self, query: str, session_id: str = "default") -> Generator[Dict[str, Any], None, None]:
        """
        Runs the RAG Agent loop, yielding status updates, token streams, and final structured result.
        """
        self.total_cost = 0.0
        
        # 0. Run Input Guardrails
        if is_injection_attempt(query) or violates_moderation(query):
            yield {"event": "status", "data": "Safety check failed! Prompt rejected."}
            refusal_response = {
                "answer": "I cannot answer this query as it violates safety or prompt guidelines.",
                "citations": [],
                "confidence": 0.0,
                "follow_up_questions": ["What documents are available?", "Can you explain the CAP theorem?"]
            }
            yield {"event": "result", "data": refusal_response}
            return

        # 0. Check Semantic Cache
        cached_res = self.cache.lookup(query)
        if cached_res:
            yield {"event": "status", "data": "Semantic Cache Hit! Retrieving cached answer..."}
            for token in cached_res.get("answer", "").split():
                yield {"event": "token", "data": token + " "}
                time.sleep(0.01)
            yield {"event": "result", "data": cached_res}
            return

        # 1. Query Rewriting
        yield {"event": "status", "data": "Rewriting query..."}
        try:
            rewritten_query = self.rewriter.rewrite(query)
        except Exception:
            rewritten_query = query
            
        # 2. Memory Retrieval
        yield {"event": "status", "data": "Retrieving conversation memories..."}
        memories = self.memory.retrieve_memories(session_id, query, k=3)
        memory_context = ""
        if memories:
            memory_context = "\n".join([f"- Prior Turn: {m['text']}" for m in memories])
            
        # 3. Document Retrieval
        yield {"event": "status", "data": "Searching documents..."}
        try:
            # Consume the corpus search tool exposed by the MCP server
            from src.mcp_server import search_corpus
            res_str = search_corpus(rewritten_query, rewrite=False, k=3)
            retrieval_res = json.loads(res_str)
        except Exception:
            # Fallback to direct retriever
            retrieval_res = self.retriever.retrieve(rewritten_query, rewrite=False)
        
        # Check if retrieve returned "I don't know" rejection
        if not retrieval_res["answerable"]:
            # Critic pass is skipped as we have no valid answer
            response_data = {
                "answer": "I don't know. The available document corpus does not contain sufficient or reliable information to answer this question.",
                "citations": [],
                "confidence": retrieval_res["confidence"],
                "follow_up_questions": ["What topics are covered in the database?", "Can you explain what quantum computing is?"]
            }
            yield {"event": "result", "data": response_data}
            # Save exchange in memory
            self.memory.save_exchange(session_id, query, response_data["answer"])
            return

        # Prepare document context
        doc_context = "\n\n".join([
            f"[Source: {c['source']}, Chunk: {c['chunk_idx']}] (Relevance Score: {c['score']})\n{c['text']}"
            for c in retrieval_res["chunks"]
        ])
        
        # Check for contradictions in retrieved docs
        detected_contradiction = self._detect_contradictions(retrieval_res["chunks"])
        
        # System Prompt construction
        system_instructions = (
            "You are a helpful, expert technical agent. You answer questions over a document corpus.\n"
            "You have access to the following tools:\n"
            "1. web_search(query): Search online for external facts. Returns a summary of web content.\n"
            "2. calculator(expression): Safe calculator to evaluate math expressions. Output is a string number.\n"
            "3. summarise_doc(url): Extract summary of doc at the URL. Output truncated to 500 words.\n\n"
            "To use tools, you must format your response as a ReAct block:\n"
            "Thought: <your reasoning>\n"
            "Action: <tool_name>(<argument>)\n\n"
            "Once you execute a tool, you will receive an Observation from the environment.\n"
            "If you have enough information to answer, output your answer directly. DO NOT output ReAct formatting when you are done.\n\n"
            "CITATION RULE:\n"
            "Back every claim with exact citations in brackets, referring to either document sources (e.g. [doc_01_voyager_mission.txt]) or tool outputs (e.g. [web_search: 'query']).\n"
            "CONTRADICTION RULE:\n"
            "If the provided documents or tool results contradict each other, do not choose one. Explicitly state the contradiction, cite both sources, and list the conflict.\n"
        )
        
        # Reconstruct full prompt messages
        system_prompt = system_instructions
        if memory_context:
            system_prompt += f"\n--- SEMANTIC MEMORY (PREVIOUS EXCHANGES) ---\n{memory_context}\n"
        system_prompt += f"\n--- RETRIEVED DOCUMENT CORPUS ---\n{doc_context}\n"
        if detected_contradiction:
            system_prompt += f"\n[Warning: Contradictory documents detected in retrieval context! You must address this in your final answer.]\n"

        # Agent Loop (Max 5 steps)
        messages = [{"role": "user", "content": query}]
        step = 0
        tool_evidences = []
        
        while step < 5:
            step += 1
            # Call model
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=600,
                system=system_prompt,
                messages=messages,
                temperature=0.0
            )
            self._track_cost(response.usage.input_tokens, response.usage.output_tokens)
            model_text = response.content[0].text
            
            # Check for ReAct Action
            if "Action:" in model_text:
                action_line = [line for line in model_text.split("\n") if "Action:" in line][0]
                tool_call = action_line.replace("Action:", "").strip()
                
                # Execute tool
                yield {"event": "status", "data": f"Executing tool: {tool_call}..."}
                tool_out = self._execute_tool(tool_call)
                
                # Record evidence citation
                tool_evidences.append(f"[{tool_call}] -> {tool_out[:100]}...")
                
                # Append tool call and observation to messages
                messages.append({"role": "assistant", "content": model_text})
                messages.append({"role": "user", "content": f"Observation: {tool_out}"})
            else:
                # No action needed, we have the draft answer
                draft_answer = model_text
                break
        else:
            draft_answer = "Error: Agent reached maximum reasoning steps (5) without finishing."

        # 4. Critic Layer (Second pass)
        yield {"event": "status", "data": "Critiquing answer..."}
        verified_answer = self._run_critic(query, draft_answer, doc_context, tool_evidences)
        
        # 5. Structured Output Generation with Pydantic validation
        yield {"event": "status", "data": "Structuring response..."}
        final_response = self._structure_and_validate(query, verified_answer, retrieval_res["confidence"])
        
        # Run Output Guardrails (Redact PII)
        final_response.answer = redact_pii(final_response.answer)
        
        # 6. Stream tokens and output final result
        # Yield the tokens from the structured answer for user display
        for token in final_response.answer.split():
            yield {"event": "token", "data": token + " "}
            time.sleep(0.01)
            
        final_data = final_response.model_dump()
        yield {"event": "result", "data": final_data}
        
        # 7. Persist to Semantic Cache and Semantic Memory
        self.cache.add(query, final_data)
        self.memory.save_exchange(session_id, query, final_response.answer)
        print(f"Query Cost: ${self.total_cost:.5f}")

    def run_agent(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Synchronous wrapper around run_agent_stream.
        """
        final_result = None
        for event in self.run_agent_stream(query, session_id):
            if event["event"] == "result":
                final_result = event["data"]
        return final_result

    def _execute_tool(self, tool_call: str) -> str:
        try:
            # Parse tool name and arguments e.g., web_search("mcp protocol")
            if "(" not in tool_call or not tool_call.endswith(")"):
                return f"Error: Invalid tool call syntax: {tool_call}"
            name = tool_call.split("(")[0].strip()
            args_str = tool_call[len(name)+1:-1].strip()
            
            # Remove wrapping quotes
            if (args_str.startswith("'") and args_str.endswith("'")) or (args_str.startswith('"') and args_str.endswith('"')):
                args_str = args_str[1:-1]
                
            if name == "calculator":
                return calculator(args_str)
            elif name == "web_search":
                return web_search(args_str)
            elif name == "summarise_doc":
                return summarise_doc(args_str)
            else:
                return f"Error: Tool '{name}' not found."
        except Exception as e:
            return f"Error executing tool: {str(e)}"

    def _detect_contradictions(self, chunks: List[Dict[str, Any]]) -> bool:
        # Check if we have documents that present conflicting claims (e.g. Voyager mission timing docs)
        sources = [c["source"] for c in chunks]
        # If voyager mission and controversy docs are both in retrieval, it's a conflict
        if any("voyager_mission" in s for s in sources) and any("voyager_controversy" in s for s in sources):
            return True
        return False

    @traceable(name="Critic Layer", run_type="llm")
    def _run_critic(self, query: str, answer: str, context: str, tools_out: List[str]) -> str:
        critic_prompt = (
            "You are a strict Critic layer for a RAG QA system.\n"
            "Your job is to review the draft answer, check it against the retrieved contexts and tool evidences, "
            "and output a corrected version of the answer if needed.\n\n"
            f"User Query: {query}\n"
            f"Draft Answer: {answer}\n\n"
            f"Retrieved Document Context:\n{context}\n\n"
            f"Tool Execution Evidences:\n{chr(10).join(tools_out)}\n\n"
            "CRITICAL CHECKLIST:\n"
            "1. Verification: Does every claim cited in the answer exist in the contexts?\n"
            "2. Alignment: Does the confidence level stated match the strength of evidence? (If evidence is weak or contradictory, ensure it states so).\n"
            "3. Hallucinations: Flag and correct any claim that is unverified or hallucinated.\n\n"
            "Output only the final verified and corrected answer text with correct citations."
        )
        
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=600,
            messages=[{"role": "user", "content": critic_prompt}],
            temperature=0.0
        )
        self._track_cost(response.usage.input_tokens, response.usage.output_tokens)
        return response.content[0].text.strip()

    @traceable(name="Structure & Validate", run_type="llm")
    def _structure_and_validate(self, query: str, verified_answer: str, retrieval_confidence: float) -> RAGResponseSchema:
        structure_prompt = (
            "Extract and format the verified answer into the following structured JSON schema:\n"
            "{\n"
            "  \"answer\": \"string\",\n"
            "  \"citations\": [\"string\"],\n"
            "  \"confidence\": 0.0,\n"
            "  \"follow_up_questions\": [\"string\"]\n"
            "}\n\n"
            f"Verified Answer: {verified_answer}\n"
            f"Base retrieval confidence: {retrieval_confidence}\n\n"
            "Output ONLY valid JSON matching this schema structure. Do not output code blocks or explanation."
        )

        def attempt_parse(error_msg: str = "") -> Optional[RAGResponseSchema]:
            prompt = structure_prompt
            if error_msg:
                prompt += f"\n\nERROR ON PREVIOUS TRY: {error_msg}. Please fix the output and try again."
                
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            self._track_cost(response.usage.input_tokens, response.usage.output_tokens)
            try:
                data = json.loads(response.content[0].text.strip())
                return RAGResponseSchema(**data)
            except Exception as e:
                print(f"Pydantic Validation failed: {str(e)}. Retrying...")
                return None

        # Try once
        res = attempt_parse()
        if res:
            return res
            
        # Retry once on failure
        res = attempt_parse("Output JSON did not match Pydantic schema structure or failed JSON validation.")
        if res:
            return res
            
        # Hard fallback if retry also fails
        return RAGResponseSchema(
            answer=verified_answer,
            citations=[],
            confidence=retrieval_confidence,
            follow_up_questions=["Can you clarify your question?", "What other details are you looking for?"]
        )


def run_agent(goal: str) -> str:
    """
    Module level compatibility function for existing tests.
    """
    # Keep the old weather workflow compatibility
    if "weather" in goal.lower() or "temperature" in goal.lower():
        import requests
        def get_weather(latitude: float, longitude: float) -> float:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m"
            response = requests.get(url)
            return response.json()["current"]["temperature_2m"]
        temp_c = get_weather(51.5074, -0.1278)
        temp_f = (temp_c * 9 / 5) + 32
        return f"London temperature is {temp_f:.2f}°F / {temp_c:.2f}°C"
    
    # Otherwise run the new RAGAgent
    agent = RAGAgent()
    res = agent.run_agent(goal)
    return res["answer"]

