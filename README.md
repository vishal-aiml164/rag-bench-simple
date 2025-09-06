# 📚 RAG-Bench

**RAG-Bench** is a modular benchmarking framework for **Retrieval-Augmented Generation (RAG)** experiments.  
It provides flexible pipelines for running, evaluating, and comparing different retrievers, RAG variants, and evaluation strategies — all configurable via YAML.

---

## ✨ Features

- 🔍 **Retrievers**
  - FAISS
  - BM25
  - Hybrid (dense + sparse)
  - Qdrant

- 🧩 **RAG Variants**
  - Naive RAG
  - Corrective RAG
  - Speculative RAG
  - Graph RAG
  - LightRAG

- 📏 **Evaluators**
  - **IR Evaluator**: classic information retrieval metrics (recall, precision, MRR).
  - **RAGAS Embedding Evaluator**: evaluates context & answer relevancy using embeddings.
  - **RAGAS LLM Evaluator**: uses an LLM-as-a-judge (faithfulness, factuality, coherence, fluency).

- ⚡ **Config-driven experiments**  
  All runs are defined in `config/experiments.yaml` and reproducible with a single command.

- 📊 **Visualization utilities**  
  Automatically generate plots of evaluation metrics for comparison.

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/rag-bench.git
cd rag-bench

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

# Install dependencies
pip install -r requirements.txt


🚀 Usage

Prepare your dataset
Place datasets under data/input/. For example:

data/input/squad/


Configure experiment
Edit config/experiments.yaml:

dataset:
  name: squad
  subset: validation

experiment:
  retriever: faiss
  rag_variant: naive_rag
  embeddings: all-MiniLM-L6-v2
  num_records: 10
  top_k: 5
  chunk_size: 512
  chunk_overlap: 0


Run an experiment

python main.py


View results
Plots and results will be saved under data/results/:

data/results/squad_validation_512_0_results.png

📊 Example Output

Average IR metrics (recall, precision, etc.)

RAGAS embedding scores (context relevancy, answer relevancy)

Optional LLM-judge scores (faithfulness, coherence, fluency)

Visual comparison plots

🛠️ Roadmap

 Add HuggingFace vLLM and local GPU LLM inference support.

 Extend to multi-hop QA datasets.

 Add grid-search runner for hyperparameter sweeps.

 Dockerize setup for easy deployment.

 Add Streamlit UI for interactive exploration.

📜 License

MIT License. See LICENSE
 for details.

🤝 Contributing

Contributions are welcome!
Please open issues for bugs/feature requests and submit PRs for enhancements.
