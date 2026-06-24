import os
import pytest
from PIL import Image
from pathlib import Path
from src.config import CORPUS_DIR
from src.indexer import run_indexing
from src.vector_store import ChromaVectorStore

@pytest.fixture
def temp_image():
    # Setup: Create a temporary image in the corpus directory
    temp_img_path = CORPUS_DIR / "test_diagram.png"
    
    # Create a simple red 100x100 pixel image
    img = Image.new('RGB', (100, 100), color='red')
    img.save(temp_img_path)
    
    yield temp_img_path
    
    # Teardown: Clean up the image
    if temp_img_path.exists():
        temp_img_path.unlink()

def test_multimodal_indexing(temp_image):
    # Retrieve current document count
    vector_store = ChromaVectorStore()
    initial_count = vector_store.get_collection_count()
    
    # Run the indexing pipeline which should capture the new image
    run_indexing(reset=False)
    
    new_count = vector_store.get_collection_count()
    
    # Verify that at least one new chunk was added to the corpus database
    assert new_count > initial_count
    
    # Query for the image content
    results = vector_store.collection.get(
        where={"source": "test_diagram.png"}
    )
    
    assert len(results["ids"]) > 0
    doc = results["documents"][0].lower()
    assert "architectural diagrams" in doc or "red" in doc
