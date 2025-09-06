import os
import ast
import time
import yaml
import numpy as np
from sentence_transformers import SentenceTransformer, util

from retrievers.faiss_retriever import FAISSRetriever
from rag_variants.naive_rag import NaiveRAG
from evaluators.ir_evaluator import IREvaluator
from utils.data_loader import get_dataset
from utils.logger import get_logger
from utils.visualizer import visualize_scores  # <-- now properly imported

logger = get_logger("Main")

class WorkingRagasEmbeddingEvaluator:
    """Mock Ragas embedding evaluator using cosine similarity."""
    def __init__(self, model):
        self.model = model

    def evaluate(self, query, answer, retrieved_docs, ground_truth):
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # Context relevancy
        doc_texts = [doc['text'] for doc in retrieved_docs]
        doc_embeddings = self.model.encode(doc_texts, convert_to_tensor=True)
        context_relevancy_score = util.cos_sim(query_embedding, doc_embeddings).mean().item()

        # Answer relevancy
        answer_embedding = self.model.encode(answer, convert_to_tensor=True)
        answer_relevancy_score = util.cos_sim(query_embedding, answer_embedding).item()

        return {
            'context_relevancy': context_relevancy_score,
            'answer_relevancy': answer_relevancy_score
        }

# ------------------------
# Dataset helpers
# ------------------------
def prepare_documents(df, dataset_name):
    if df is None:
        return []
    if dataset_name == "squad":
        return [{"id": str(row.name), "text": row["context"]} for _, row in df.iterrows()]
    elif "sentence-transformers/msmarco" in dataset_name:
        return [{"id": str(row.name), "text": row["text"]} for _, row in df.iterrows()]
    elif dataset_name == "msmarco":
        documents = []
        for _, row in df.iterrows():
            try:
                passages_data = ast.literal_eval(row['passages'])
                for text in passages_data['passage_text']:
                    documents.append({"id": f"{row.name}-{len(documents)}", "text": text})
            except Exception:
                continue
        return documents
    elif dataset_name == "natural_questions":
        return [{"id": str(i), "text": row["document_text"]} for i, row in df.iterrows()]
    elif dataset_name == "beir":
        col = "text" if "text" in df.columns else "document"
        return [{"id": str(i), "text": row[col]} for i, row in df.iterrows()]
    elif dataset_name == "wikitext":
        return [{"id": str(i), "text": row["text"]} for i, row in df.iterrows()]
    else:
        raise ValueError(f"No document conversion rule for dataset {dataset_name}")

def extract_ground_truth(row, dataset_name):
    if dataset_name == "squad":
        answers = row.get("answers", {}).get("text", [])
        if isinstance(answers, np.ndarray):
            answers = answers.tolist()
        return answers[0] if answers else None
    elif dataset_name == "msmarco":
        return row.get("answer_text")
    elif dataset_name == "natural_questions":
        return row.get("short_answer")
    elif dataset_name == "beir":
        return row.get("answer")
    elif dataset_name == "wikitext":
        return row["text"][1:] if "text" in row else None
    return None

# ------------------------
# Main pipeline
# ------------------------
def main():
    logger.info("Pipeline started")
    start = time.time()

    # Load config
    with open("config/experiments.yaml", "r") as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]
    subset_name = config["dataset"]["subset"]
    exp = config["experiment"]  # <-- define exp from config
    chunk_size = exp.get("chunk_size", 512)
    chunk_overlap = exp.get("chunk_overlap", 0)

    run_id = f"{dataset_name}_{subset_name}_{chunk_size}_{chunk_overlap}"

    # Load embedding model
    model = SentenceTransformer(exp["embeddings"])
    embedding_dim = model.get_sentence_embedding_dimension()

    # Load dataset
    dataframes = get_dataset(dataset_name=dataset_name, data_dir="data/input")
    val_df = dataframes.get(subset_name)
    if val_df is None:
        logger.error(f"No validation data found for {dataset_name}")
        return

    # Build / load FAISS retriever
    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, f"{dataset_name}_{subset_name}_index.bin")
    doc_path = os.path.join(output_dir, f"{dataset_name}_{subset_name}_docs.json")

    retriever = FAISSRetriever.load_or_build(
        run_id,
        embedding_dim,
        model.encode,
        prepare_documents(val_df, dataset_name)
    )


    # Init RAG
    rag = NaiveRAG(retriever)

    # Evaluators
    ir_eval = IREvaluator()
    ragas_embed_eval = WorkingRagasEmbeddingEvaluator(model)

    ir_scores_list, ragas_embed_scores_list = [], []
    num_records = exp.get("num_records", 10)
    top_k = exp.get("top_k", 5)

    for _, row in val_df.head(num_records).iterrows():
        query = row["question"]
        ground_truth = extract_ground_truth(row, dataset_name)

        retrieved_docs = retriever.retrieve(query, top_k=top_k)
        answer = "This is a placeholder answer."

        ir_scores = ir_eval.evaluate(query, retrieved_docs, ground_truth)
        ragas_embed_scores = ragas_embed_eval.evaluate(query, answer, retrieved_docs, ground_truth)

        ir_scores_list.append(ir_scores)
        ragas_embed_scores_list.append(ragas_embed_scores)

    # Average scores
    if ir_scores_list:
        ir_keys = set().union(*ir_scores_list)
        avg_ir_scores = {k: np.mean([d[k] for d in ir_scores_list if k in d]) for k in ir_keys}
    else:
        avg_ir_scores = {}

    if ragas_embed_scores_list:
        ragas_keys = set().union(*ragas_embed_scores_list)
        avg_ragas_embed_scores = {k: np.mean([d[k] for d in ragas_embed_scores_list if k in d]) for k in ragas_keys}
    else:
        avg_ragas_embed_scores = {}

    print("\n📊 Average IR Scores:", avg_ir_scores)
    print("\n📊 Average RAGAS Embedding Scores:", avg_ragas_embed_scores)

    # Visualization
    results_dir = "data/results"
    os.makedirs(results_dir, exist_ok=True)
    #filename = os.path.join(results_dir, f"{exp['name']}_results.png")

        # Save visualization
    filename = os.path.join(results_dir, f"{run_id}_results.png")
    #visualize_scores(avg_ir_scores, avg_ragas_scores, filename)

    if avg_ir_scores or avg_ragas_embed_scores:
        visualize_scores(avg_ir_scores, avg_ragas_embed_scores, filename)

    logger.info(f"Experiment {run_id} finished in {time.time() - start:.2f} sec")

if __name__ == "__main__":
    main()
