# utils/memory_estimator.py
import os
import json
import time
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

def estimate_chunks_and_memory(
    dataset_name="wikitext", 
    subset="wikitext-2-raw-v1", 
    split="train", 
    chunk_size=512, 
    overlap=10, 
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    dtype=np.float32
):
    """
    Estimates memory + time required for embeddings and FAISS index.
    """

    print(f"Loading dataset: {dataset_name}/{subset} [{split}] ...")
    dataset = load_dataset(dataset_name, subset, split=split)

    print("Counting total tokens...")
    texts = dataset["text"]
    total_tokens = sum(len(t.split()) for t in texts)  # rough tokenization by whitespace

    # Estimate number of chunks
    effective_size = chunk_size - overlap
    num_chunks = total_tokens // effective_size
    print(f"Total tokens: {total_tokens:,}")
    print(f"Chunk size: {chunk_size}, Overlap: {overlap}")
    print(f"Estimated number of chunks: {num_chunks:,}")

    # Embedding model
    model = SentenceTransformer(model_name)
    embedding_dim = model.get_sentence_embedding_dimension()
    bytes_per_value = np.dtype(dtype).itemsize

    # Memory usage
    bytes_per_embedding = embedding_dim * bytes_per_value
    total_mem = num_chunks * bytes_per_embedding  # embeddings only
    total_mem_MB = total_mem / (1024**2)
    total_mem_GB = total_mem / (1024**3)

    # Add FAISS overhead ~ 50%
    total_mem_with_index = total_mem * 1.5
    total_mem_with_index_GB = total_mem_with_index / (1024**3)

    print(f"\nEmbedding dimension: {embedding_dim}")
    print(f"Per embedding size: {bytes_per_embedding} bytes")
    print(f"Total embedding memory: {total_mem_MB:.2f} MB (~{total_mem_GB:.2f} GB)")
    print(f"With FAISS index: ~{total_mem_with_index_GB:.2f} GB")

    # Rough time estimate (encode 1000 chunks)
    sample_texts = ["This is a sample sentence."] * 1000
    start = time.time()
    _ = model.encode(sample_texts, show_progress_bar=True)
    end = time.time()
    time_per_1k = end - start
    est_time_total = (num_chunks / 1000) * time_per_1k

    print(f"\nTime for 1000 encodings: {time_per_1k:.2f} sec")
    print(f"Estimated total embedding time: {est_time_total/60:.1f} minutes (~{est_time_total/3600:.1f} hours)")

if __name__ == "__main__":
    # Example run
    estimate_chunks_and_memory(
        dataset_name="wikitext",
        subset="wikitext-2-raw-v1",
        split="train",
        chunk_size=512,
        overlap=10,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
