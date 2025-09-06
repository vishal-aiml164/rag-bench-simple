# rag_variants/naive_rag.py
class NaiveRAG:
    def __init__(self, retriever):
        self.retriever = retriever

    def answer(self, query):
        retrieved_docs = self.retriever.retrieve(query, top_k=2)
        context = " ".join([doc["text"] for doc in retrieved_docs])
        # Simulate LLM call (for demo, just string concat)
        answer = f"Based on context: {context}"
        return answer, retrieved_docs
