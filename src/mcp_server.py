import json
from mcp.server.fastmcp import FastMCP
from src.retriever import DocumentRetriever

mcp = FastMCP("Corpus Search Server")
retriever = DocumentRetriever()

@mcp.tool()
def search_corpus(query: str, rewrite: bool = True, k: int = 3) -> str:
    """
    Search the local document corpus for passages relevant to the query.
    Optionally rewrites the query to expand abbreviations and add synonyms for better recall.
    Returns a JSON string containing matched chunks, source documents, chunk indices, and confidence scores.
    """
    try:
        results = retriever.retrieve(query, rewrite=rewrite, k=k)
        return json.dumps(results, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to search corpus: {str(e)}"}, indent=2)

if __name__ == "__main__":
    mcp.run()
