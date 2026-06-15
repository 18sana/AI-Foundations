from src.rag_system import rag_answer


def test_rag_mock(monkeypatch):

    def fake_claude(question):
        return "Mock Claude Response"

    monkeypatch.setattr(
        "src.rag_system.ask_claude",
        fake_claude
    )

    result = rag_answer(
        "What is RAG?"
    )

    assert result == "Mock Claude Response"