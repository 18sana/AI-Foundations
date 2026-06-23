import pytest
from src.memory import SemanticMemory

@pytest.fixture
def memory_store():
    # Setup test memory collection
    test_collection_name = "test_memory_collection"
    mem = SemanticMemory(collection_name=test_collection_name)
    mem.reset_memories()
    yield mem
    # Cleanup
    try:
        mem.client.delete_collection(test_collection_name)
    except Exception:
        pass

def test_store_and_retrieve_basic(memory_store):
    # Test 1: Store memory. Retrieve it. Should return successfully.
    session_id = "session_basic"
    user_msg = "My favorite color is green."
    assistant_msg = "That's nice! Green is a calming color."
    
    memory_store.save_exchange(session_id, user_msg, assistant_msg)
    
    # Retrieve using a query semantically close
    results = memory_store.retrieve_memories(session_id, "What color do I like?", k=1)
    
    assert len(results) == 1
    assert "favorite color is green" in results[0]["text"]
    assert results[0]["user"] == user_msg
    assert results[0]["assistant"] == assistant_msg

def test_semantic_preference_retrieval(memory_store):
    # Test 2: Store model preference, query which model to use, retrieve preference memory
    session_id = "session_pref"
    
    # Store unrelated exchanges
    memory_store.save_exchange(session_id, "How is the weather today?", "It is sunny outside.")
    memory_store.save_exchange(session_id, "I prefer using Claude over GPT.", "Got it, you like Claude.")
    memory_store.save_exchange(session_id, "What is 2 + 2?", "2 + 2 is 4.")
    
    # Query for the preference
    results = memory_store.retrieve_memories(session_id, "Which model should I use?", k=1)
    
    assert len(results) == 1
    assert "Claude over GPT" in results[0]["text"]

def test_top_3_returned(memory_store):
    # Test 3: Store multiple memories. Ensure Top 3 are returned.
    session_id = "session_top3"
    
    exchanges = [
        ("I like apples.", "Apples are delicious."),
        ("I enjoy playing tennis.", "Tennis is a great sport."),
        ("I have a pet dog.", "Dogs are loyal companions."),
        ("My office is in Seattle.", "Seattle is a nice city."),
        ("I drive a blue car.", "Blue cars look great.")
    ]
    
    for user, assistant in exchanges:
        memory_store.save_exchange(session_id, user, assistant)
        
    # Query asking for 3 memories
    results = memory_store.retrieve_memories(session_id, "Tell me about my hobbies and likes", k=3)
    
    assert len(results) == 3
    # Ensure they are sorted by relevance
    assert results[0]["score"] >= results[1]["score"]
    assert results[1]["score"] >= results[2]["score"]

def test_session_isolation(memory_store):
    # Test 4: Session Isolation (Session A queries never retrieve Session B memories)
    session_a = "session_a"
    session_b = "session_b"
    
    # Save different preferences
    memory_store.save_exchange(session_a, "I prefer using Claude over GPT.", "Okay, Claude.")
    memory_store.save_exchange(session_b, "I prefer using GPT over Claude.", "Okay, GPT.")
    
    # Query from Session A
    results_a = memory_store.retrieve_memories(session_a, "Which model should I use?", k=2)
    assert len(results_a) == 1
    assert "Claude over GPT" in results_a[0]["text"]
    assert "GPT over Claude" not in results_a[0]["text"]
    
    # Query from Session B
    results_b = memory_store.retrieve_memories(session_b, "Which model should I use?", k=2)
    assert len(results_b) == 1
    assert "GPT over Claude" in results_b[0]["text"]
    assert "Claude over GPT" not in results_b[0]["text"]
