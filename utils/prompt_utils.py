SYSTEM_PROMPT = """You are a careful, terse assistant. Cite snippets from the provided context explicitly."""

def build_prompt(question: str, contexts: list[str]) -> str:
    ctx = "\n\n".join([f"[{i}] {c}" for i, c in enumerate(contexts, 1)])
    return f"""{SYSTEM_PROMPT}

Question:
{question}

Context:
{ctx}

Answer succinctly and cite [#] where relevant."""
