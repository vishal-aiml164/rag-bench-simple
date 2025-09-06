from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, tokenized_docs: list[list[str]]):
        self.bm25 = BM25Okapi(tokenized_docs)
        self.ids = list(range(len(tokenized_docs)))

    def search(self, query_tokens: list[str], top_k: int):
        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(zip(self.ids, scores), key=lambda x: -x[1])[:top_k]
        return ranked
