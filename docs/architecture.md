# Architecture

```text
Query -> Retriever (BM25 / FAISS / Qdrant / Hybrid) -> Candidate Contexts
      -> RAG Variant (Naive / Corrective / Speculative / Graph / Light)
      -> LLM Answer
      -> Evaluators: Traditional (Recall/MRR/NDCG), RAGAS, LLM-as-Judge
      -> Metrics + Visualizations
```

rag-bench/
│
├── README.md                  # Overview of the project and usage
├── LICENSE
├── requirements.txt           # All dependencies
├── config/
│   ├── experiments.yaml       # Config for each experiment run
│   └── retriever_config.yaml  # Vector store / BM25 / hybrid settings
│
├── data/
│   ├── input/                 # Datasets (CUAD, HotpotQA, etc.)
│   └── output/                # Raw results, logs
│
├── rag_variants/
│   ├── naive_rag.py
│   ├── corrective_rag.py
│   ├── speculative_rag.py
│   ├── graph_rag.py
│   └── lightrag.py
│
├── retrievers/
│   ├── faiss_retriever.py
│   ├── bm25_retriever.py
│   ├── hybrid_retriever.py
│   └── qdrant_retriever.py
│
├── evaluators/
│   ├── ragas_eval.py
│   ├── llm_judge_eval.py
│   └── traditional_metrics.py
│
├── runners/
│   ├── run_experiment.py      # Main entry point for experiments
│   └── search_grid.py         # Grid/Bayesian param tuning
│
├── utils/
│   ├── logger.py
│   ├── prompt_utils.py
│   └── visualizer.py
│
├── notebooks/
│   └── analysis.ipynb         # Visualize experiment results
└── docs/
    └── architecture.md        # Architecture diagram and flow
