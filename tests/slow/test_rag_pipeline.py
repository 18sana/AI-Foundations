# pyrefly: ignore [missing-import]
from src.rag import rag_answer


def test_rag_pipeline():

    answer = rag_answer(
        "What is RAG?"
    )

    assert "retrieval" in answer.lower()