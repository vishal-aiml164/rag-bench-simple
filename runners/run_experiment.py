# runners/run_experiment.py
import yaml
import numpy as np
from retrievers.faiss_retriever import FAISSRetriever
from rag_variants.naive_rag import NaiveRAG

# ---- Dummy LLM Stub ----
def mock_llm(prompt: str) -> str:
    return "This is a dummy answer based on retrieved context."

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    # load config
    config = load_config("config/experiments.yaml")
    dim = config["retriever"]["embedding_dim"]

    # initialize retriever
    retriever = FAISSRetriever(dim=dim, k=config["retriever"]["top_k"])

    # simulate dataset
    docs = [
        "The mitochondria is the powerhouse of the cell.",
        "Paris is the capital of France.",
        "The Earth revolves around the Sun.",
        "Python is a popular programming language.",
        "The stock market fluctuates daily."
    ]
    embeddings = np.random.rand(len(docs), dim).astype("float32")  # dummy embeddings
    retriever.add(embeddings, docs)

    # setup RAG
    rag = NaiveRAG(retriever, mock_llm)

    # query
    query = "What is the capital of France?"
    query_embedding = np.random.rand(1, dim).astype("float32")
    answer = rag.generate(query, query_embedding)

    print(f"Query: {query}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
