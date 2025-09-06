# Speculative RAG: draft lightweight answer, retrieve more if low confidence, refine.
def answer(question: str, contexts: list[str]) -> dict:
    return {"answer": "<speculative answer>", "notes": "confidence gating not implemented"}
