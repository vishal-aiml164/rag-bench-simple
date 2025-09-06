from rank_bm25 import BM25Okapi
from sklearn.metrics import precision_score, recall_score, f1_score

class IREvaluator:
    def __init__(self):
        pass

    def evaluate(self, query, retrieved_docs, ground_truth):
        """
        Evaluates retrieval quality.
        query: user query string
        retrieved_docs: list of dicts [{"id": .., "text": ..}, ...]
        ground_truth: string or list of relevant doc ids
        """

        # Here we simplify: check if ground_truth substring exists in any doc
        retrieved_texts = [doc["text"] for doc in retrieved_docs]
        relevant = [1 if ground_truth in text else 0 for text in retrieved_texts]

        # For demonstration, assume first doc should be relevant
        y_true = [1] + [0]*(len(relevant)-1)
        y_pred = relevant

        return {
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)
        }
