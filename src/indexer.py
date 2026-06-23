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
    
    corpus_files = sorted(list(CORPUS_DIR.glob("*.txt")))
    print(f"Found {len(corpus_files)} files in corpus.")
    
    for filepath in corpus_files:
        filename = filepath.name
        content = filepath.read_text(encoding="utf-8")
        
        chunks = chunk_text(content)
        print(f"Processing '{filename}': split into {len(chunks)} chunks.")
        
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
