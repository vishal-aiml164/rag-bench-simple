# rag-bench

Benchmark and compare RAG variants (naive, corrective, speculative, graph/light RAG) across datasets like FinQA, CommonWikiText, PubMedQA, BioASQ.
It provides:
- Pluggable retrievers (BM25, FAISS, Qdrant, Hybrid)
- Config-driven experiments
- Traditional retrieval metrics (Recall@k, MRR, NDCG) and optional RAGAS
- Grid / Random / Bayesian (Gaussian Process) hyperparameter search

## Quick start
```bash
pip install -r requirements.txt
python -m runners.run_experiment --exp config/experiments.yaml
```

See `docs/architecture.md` for the flow and `config/*.yaml` for settings.
