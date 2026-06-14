from src.claude_client import ask_claude


def rag_answer(question):
    return ask_claude(question)