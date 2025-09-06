# Thin wrapper to compute RAGAS metrics if you have answers and contexts.
from typing import List, Dict, Any
try:
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from ragas import evaluate
except Exception:
    faithfulness = answer_relevancy = context_precision = None
    evaluate = None

def eval_ragas(samples: List[Dict[str, Any]]):
    if evaluate is None:
        raise ImportError("ragas is not installed or failed to import.")
    return evaluate(samples, metrics=[faithfulness, answer_relevancy, context_precision])
