import os
import time
from ragas import evaluate
from ragas.metrics import answer_relevancy
from datasets import Dataset

class RagasEmbeddingEvaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, query, answer, retrieved_docs, ground_truth):
        """
        Evaluates RAG pipeline using Ragas embedding-based metrics.
        """
        data_samples = {
            'question': [query],
            'answer': [answer],
            'contexts': [[doc['text'] for doc in retrieved_docs]],
            'ground_truth': [ground_truth]
        }
        dataset = Dataset.from_dict(data_samples)

        # Ragas will use the embedding model you've provided
        # for these metrics, avoiding the need for an LLM API key.
        metrics_to_evaluate = [answer_relevancy]

        try:
            result = evaluate(
                dataset,
                metrics=metrics_to_evaluate,
                embeddings=self.model
            )
            # The result object is a Dataset, we'll convert it to a dictionary
            # for easy use in the main script.
            result_df = result.to_pandas()
            return result_df.to_dict(orient="records")[0] if not result_df.empty else {}
        except Exception as e:
            print(f"Error during Ragas evaluation: {e}")
            return {}
