import os
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

class QueryRewriter:
    def __init__(self, model_name: str = "claude-haiku-4-5"):
        load_dotenv()
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = Anthropic(api_key=self.api_key)
        self.model_name = model_name

    @traceable(name="Query Rewriter", run_type="llm")
    def rewrite(self, query: str) -> str:
        """
        Rewrites a user's question to make it optimized for a vector database.
        Expands acronyms, adds search terms, and outputs only the search keywords.
        """
        system_prompt = (
            "You are a search query optimizer for a Vector Database. "
            "Your job is to rewrite the user's input query into a search query that maximizes retrieval recall. "
            "Follow these rules strictly:\n"
            "1. Translate abbreviations to their full form (e.g. 'MCP' becomes 'Model Context Protocol').\n"
            "2. Add relevant search keywords, synonyms, and context terms.\n"
            "3. Keep it brief and focused on retrieval keywords.\n"
            "4. Output ONLY the optimized search query. DO NOT provide conversational introductions, explanations, or answers to the user's question.\n"
            "5. Do not put quotes around the query."
        )

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=100,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": query
                }
            ],
            temperature=0.0
        )
        
        rewritten = response.content[0].text.strip()
        # Strip wrapping quotes if any
        if rewritten.startswith('"') and rewritten.endswith('"'):
            rewritten = rewritten[1:-1].strip()
        return rewritten
