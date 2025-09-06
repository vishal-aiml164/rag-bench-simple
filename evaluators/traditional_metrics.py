# evaluators/traditional_metrics.py
class SimpleEvaluator:
    def evaluate(self, query, answer, ground_truth):
        return 1.0 if ground_truth.lower() in answer.lower() else 0.0
