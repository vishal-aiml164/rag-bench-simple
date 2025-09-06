from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate
import pandas as pd

class RagasLLMEvaluator:
    def __init__(self, model="gpt-4"):
        """
        model: str
          "gpt-4", "gpt-3.5-turbo" (paid OpenAI)
          OR local OSS model served with vLLM / HF pipeline
        """
        self.model = model

    def evaluate(self, query, answer, ground_truth):
        """
        Paid / GPU mode RAGAS evaluation with LLM-as-a-judge.
        """
        df = pd.DataFrame([{
            "question": query,
            "answer": answer,
            "contexts": [ground_truth]
        }])

        results = evaluate(
            df,
            metrics=[faithfulness, answer_relevancy],
            llm=self.model
        )
        return results.to_pandas().to_dict(orient="records")[0]
