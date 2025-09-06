# Corrective RAG skeleton: produce draft answer, verify against sources, correct if ungrounded.
def answer(question: str, contexts: list[str]) -> dict:
    return {"answer": "<corrected answer>", "notes": "verification+edit pass not implemented"}
