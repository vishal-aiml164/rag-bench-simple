def chunk_text(text, chunk_size, overlap):
    """
    Splits text into overlapping chunks for retrieval.
    """
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append({"id": f"{start}-{end}", "text": text[start:end]})
        start += chunk_size - overlap
    return chunks
