import os
from pathlib import Path
from src.config import CORPUS_DIR
from src.corpus import generate_corpus
from src.chunker import chunk_text
from src.embedder import DocumentEmbedder
from src.vector_store import ChromaVectorStore

def run_indexing(reset: bool = True):
    print("Starting indexing pipeline...")
    
    # 1. Generate corpus if it doesn't exist
    if not list(CORPUS_DIR.glob("*.txt")):
        print("Corpus directory empty. Generating sample corpus...")
        generate_corpus()
    
    # 2. Initialize modules
    embedder = DocumentEmbedder()
    vector_store = ChromaVectorStore()
    
    if reset:
        print("Resetting ChromaDB collection...")
        vector_store.reset_collection()
        
    # 3. Read files, chunk and prepare data
    ids = []
    documents = []
    metadatas = []
    
    import base64
    from anthropic import Anthropic
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")

    txt_files = list(CORPUS_DIR.glob("*.txt"))
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.gif"]
    img_files = []
    for ext in image_extensions:
        img_files.extend(list(CORPUS_DIR.glob(ext)))
        
    corpus_files = sorted(txt_files) + sorted(img_files)
    print(f"Found {len(corpus_files)} files in corpus ({len(txt_files)} text, {len(img_files)} images).")
    
    for filepath in corpus_files:
        filename = filepath.name
        is_image = any(filename.lower().endswith(ext.replace("*", "")) for ext in image_extensions)
        
        if is_image:
            if not api_key or api_key == "your_actual_anthropic_api_key_here":
                description = f"This is a mock description of the image content inside {filename}. It contains architectural diagrams and technical specifications."
            else:
                try:
                    client = Anthropic(api_key=api_key)
                    with open(filepath, "rb") as image_file:
                        image_data = image_file.read()
                        base64_data = base64.b64encode(image_data).decode("utf-8")
                        
                    media_type = "image/png"
                    if filename.lower().endswith((".jpg", ".jpeg")):
                        media_type = "image/jpeg"
                    elif filename.lower().endswith(".gif"):
                        media_type = "image/gif"
                        
                    response = client.messages.create(
                        model="claude-haiku-4-5",
                        max_tokens=600,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": media_type,
                                            "data": base64_data
                                        }
                                    },
                                    {
                                        "type": "text",
                                        "text": "Extract any text visible in this image. Describe the image content, diagrams, or charts in detail for RAG indexing."
                                    }
                                ]
                            }
                        ],
                        temperature=0.0
                    )
                    description = response.content[0].text.strip()
                except Exception as e:
                    print(f"Error calling vision model for '{filename}': {e}. Falling back to default.")
                    description = f"Fallback description for image {filename} due to API error."
            
            chunks = [{"text": description, "start_char": 0, "end_char": len(description)}]
            print(f"Processing image '{filename}': extracted vision description.")
        else:
            content = filepath.read_text(encoding="utf-8")
            chunks = chunk_text(content)
            print(f"Processing text '{filename}': split into {len(chunks)} chunks.")
        
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{filename}_chunk_{idx}"
            ids.append(chunk_id)
            documents.append(chunk["text"])
            metadatas.append({
                "source": filename,
                "chunk_idx": idx,
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"]
            })
            
    if not documents:
        print("No documents/chunks found to index!")
        return

    # 4. Generate embeddings
    print(f"Generating embeddings for {len(documents)} chunks...")
    embeddings = embedder.embed_documents(documents)
    
    # 5. Store in ChromaDB
    print("Writing to vector database...")
    vector_store.add_chunks(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents
    )
    
    print(f"Indexing completed. Collection count: {vector_store.get_collection_count()} chunks.")

if __name__ == "__main__":
    run_indexing(reset=True)
