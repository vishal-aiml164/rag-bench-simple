import faiss
import numpy as np
import time
import os
import json
from utils.logger import get_logger

logger = get_logger("FAISSRetriever")


class FAISSRetriever:
    def __init__(self, index, documents, embedding_dim, embed_fn, run_id):
        """
        Initializes the retriever with an existing FAISS index and documents.
        """
        self.index = index
        self.documents = documents
        self.embed_fn = embed_fn
        self.embedding_dim = embedding_dim
        self.run_id = run_id
        logger.info("FAISSRetriever initialized. Total docs=%d", len(self.documents))

    @staticmethod
    def load_or_build(run_id, embedding_dim, embed_fn, documents_to_add):
        """
        Checks for a cached index and loads it, or builds a new one and saves it.
        The filenames are automatically derived from the run_id.
        """
        output_dir = "data/output"
        os.makedirs(output_dir, exist_ok=True)

        index_path = os.path.join(output_dir, f"{run_id}_index.faiss")
        doc_path = os.path.join(output_dir, f"{run_id}_docs.json")

        if os.path.exists(index_path) and os.path.exists(doc_path):
            logger.info("Found cached index and documents. Loading from disk...")
            index = faiss.read_index(index_path)
            with open(doc_path, 'r') as f:
                documents = json.load(f)
            return FAISSRetriever(index, documents, embedding_dim, embed_fn, run_id)
        else:
            logger.info("No cached index found. Building a new one...")
            index = faiss.IndexFlatL2(embedding_dim)
            retriever = FAISSRetriever(index, [], embedding_dim, embed_fn, run_id)
            retriever.add_documents(documents_to_add)
            retriever.save()
            return retriever

    def save(self):
        """
        Saves the FAISS index and documents to the standard path derived from run_id.
        """
        output_dir = "data/output"
        os.makedirs(output_dir, exist_ok=True)

        index_path = os.path.join(output_dir, f"{self.run_id}_index.faiss")
        doc_path = os.path.join(output_dir, f"{self.run_id}_docs.json")

        logger.info("Saving FAISS index to %s", index_path)
        faiss.write_index(self.index, index_path)

        logger.info("Saving documents to %s", doc_path)
        with open(doc_path, 'w') as f:
            json.dump(self.documents, f)

        logger.info("Index and documents saved successfully.")

    def add_documents(self, documents):
        """
        documents: list of dicts with at least a "text" field
        """
        if not documents:
            logger.warning("No documents to add.")
            return

        logger.info("Adding %d documents to FAISS index...", len(documents))
        start = time.time()
        texts = [doc["text"] for doc in documents]

        # --- Ensure embeddings are numpy arrays ---
        embeddings = self.embed_fn(texts, convert_to_numpy=True)
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        # Ensure dimensions match before adding
        if embeddings.shape[1] != self.embedding_dim:
            logger.error(
                "Embedding dimension mismatch: expected %d, got %d",
                self.embedding_dim,
                embeddings.shape[1],
            )
            return

        # --- Add to FAISS index ---
        self.index.add(embeddings)

        # --- Maintain doc mapping ---
        self.documents.extend(documents)
        elapsed = time.time() - start
        logger.info(
            "Added %d docs in %.2f seconds. Total docs=%d",
            len(documents),
            elapsed,
            len(self.documents),
        )

    def retrieve(self, query, top_k=3):
        """
        query: str
        top_k: number of results
        returns: list of documents
        """
        logger.info("Retrieving top %d results for query: '%s'", top_k, query[:50])
        start = time.time()
        query_vec = self.embed_fn([query], convert_to_numpy=True)
        if not isinstance(query_vec, np.ndarray):
            query_vec = np.array(query_vec)

        # Ensure query vector dimension matches index
        if query_vec.shape[1] != self.embedding_dim:
            logger.error(
                "Query embedding dimension mismatch: expected %d, got %d",
                self.embedding_dim,
                query_vec.shape[1],
            )
            return []

        D, I = self.index.search(query_vec, top_k)

        results = []
        for idx in I[0]:
            if 0 <= idx < len(self.documents):
                results.append(self.documents[idx])

        elapsed = time.time() - start
        logger.info("Retrieved %d results in %.2f seconds", len(results), elapsed)

        return results
