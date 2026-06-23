from typing import List, Dict, Any
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """
    Splits text into chunks of size `chunk_size` with an overlap of `chunk_overlap`.
    Returns a list of dicts: [{'text': chunk_text, 'start_idx': int, 'end_idx': int}]
    """
    if not text:
        return []
    
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")
        
    chunks = []
    text_len = len(text)
    start = 0
    
    while start < text_len:
        end = start + chunk_size
        chunk_content = text[start:end]
        
        # If we are not at the end of the text, try to find a natural boundary (period, newline, space)
        # to avoid cutting off mid-word, within the last 15% of the chunk size.
        if end < text_len:
            boundary_limit = int(chunk_size * 0.15)
            search_window = chunk_content[-boundary_limit:]
            
            # Look for last sentence separator or space
            best_boundary = -1
            for separator in ["\n\n", "\n", ". ", "? ", "! ", " "]:
                idx = search_window.rfind(separator)
                if idx != -1:
                    # Adjust index relative to the full chunk
                    best_boundary = (chunk_size - boundary_limit) + idx + len(separator)
                    break
            
            if best_boundary != -1:
                end = start + best_boundary
                chunk_content = text[start:end]
        
        chunks.append({
            "text": chunk_content.strip(),
            "start_char": start,
            "end_char": end
        })
        
        start = end - chunk_overlap
        if start >= text_len or end >= text_len:
            break
            
    return chunks
