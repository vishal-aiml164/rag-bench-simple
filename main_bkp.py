import os
import ast
import time
import json
from sentence_transformers import SentenceTransformer, util
from retrievers.faiss_retriever import FAISSRetriever
from rag_variants.naive_rag import NaiveRAG
from evaluators.ir_evaluator import IREvaluator
# from evaluators.ragas_embedding_evaluator import RagasEmbeddingEvaluator
from utils.data_loader import get_dataset
from utils.logger import get_logger
import numpy as np
import matplotlib.pyplot as plt

logger = get_logger("Main")

class WorkingRagasEmbeddingEvaluator:
    """
    A working mock evaluator to demonstrate Ragas-like embedding scores.
    It calculates scores based on cosine similarity.
    """
    def __init__(self, model):
        self.model = model

    def evaluate(self, query, answer, retrieved_docs, ground_truth):
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # Calculate a mock context relevancy score
        doc_texts = [doc['text'] for doc in retrieved_docs]
        doc_embeddings = self.model.encode(doc_texts, convert_to_tensor=True)
        # Average similarity of docs to the query
        context_relevancy_score = util.cos_sim(query_embedding, doc_embeddings).mean().item()

        # Calculate a mock answer relevancy score
        answer_embedding = self.model.encode(answer, convert_to_tensor=True)
        # Similarity of the placeholder answer to the query
        answer_relevancy_score = util.cos_sim(query_embedding, answer_embedding).item()

        return {
            'context_relevancy': context_relevancy_score,
            'answer_relevancy': answer_relevancy_score
        }

def prepare_documents(df, dataset_name):
    """
    Convert dataset-specific columns into retriever documents.
    """
    if df is None:
        return []

    # Check for the specific dataset and handle its columns
    if dataset_name == "squad":
        # SQuAD has 'context' as the document text
        if 'context' in df.columns:
            return [{"id": str(row.name), "text": row["context"]} for _, row in df.iterrows()]
        else:
            raise ValueError(f"SQuAD dataset must have a 'context' column.")

    elif "sentence-transformers/msmarco" in dataset_name:
        # This dataset has a 'text' column directly
        return [{"id": str(row.name), "text": row["text"]} for _, row in df.iterrows()]

    elif dataset_name == "msmarco":
        documents = []
        for _, row in df.iterrows():
            try:
                passages_data = ast.literal_eval(row['passages'])
                passage_texts = passages_data['passage_text']
                for text in passage_texts:
                    documents.append({"id": f"{row.name}-{len(documents)}", "text": text})
            except (KeyError, ValueError, TypeError) as e:
                # Silently skip rows with malformed data
                continue
        return documents
    elif dataset_name == "natural_questions":
        return [{"id": str(i), "text": row["document_text"]} for i, row in df.iterrows()]
    elif dataset_name == "beir":
        # BEIR datasets usually have "text" or "document"
        col = "text" if "text" in df.columns else "document"
        return [{"id": str(i), "text": row[col]} for i, row in df.iterrows()]
    elif dataset_name == "wikitext":
        return [{"id": str(i), "text": row["text"]} for i, row in df.iterrows()]
    else:
        raise ValueError(f"No document conversion rule for dataset {dataset_name}")

def extract_ground_truth(row, dataset_name):
    """
    Extract dataset-specific ground truth answers for evaluation.
    """
    if dataset_name == "squad":
        # SQuAD answers are in a nested dictionary
        answers = row.get("answers", {}).get("text", [])
        # Fix: Convert to list if it's a numpy array to avoid ValueError
        if isinstance(answers, np.ndarray):
            answers = answers.tolist()
        return answers[0] if answers and len(answers) > 0 else None
    elif dataset_name == "msmarco":
        return row["answer_text"] if "answer_text" in row else None
    elif dataset_name == "natural_questions":
        return row["short_answer"] if "short_answer" in row else None
    elif dataset_name == "beir":
        # For BEIR, use qrels later; for now fallback to 'answer' col if present
        return row["answer"] if "answer" in row else None
    elif dataset_name == "wikitext":
        # Language modeling -> next token prediction
        return row["text"][1:] if "text" in row else None
    else:
        return None

def visualize_scores(ir_scores, ragas_embed_scores, filename):
    """
    Creates and saves a grouped bar chart of the evaluation scores.
    """
    print("\n📈 Creating grouped performance chart...")

    ir_metrics = list(ir_scores.keys())
    ir_values = list(ir_scores.values())

    ragas_metrics = list(ragas_embed_scores.keys())
    ragas_values = list(ragas_embed_scores.values())

    # Get all metric names for the x-axis ticks
    all_metrics = ir_metrics + ragas_metrics

    # Set up positions for grouped bars
    x_ir = np.arange(len(ir_metrics))
    x_ragas = np.arange(len(ragas_metrics)) + len(ir_metrics) + 0.5 # Add offset for a gap

    width = 0.35

    # Create the grouped bar chart
    plt.figure(figsize=(12, 7))

    bars1 = plt.bar(x_ir, ir_values, width, label='IR Scores', color='#1f77b4')
    bars2 = plt.bar(x_ragas, ragas_values, width, label='Embedding Scores', color='#2ca02c')
    
    # Add labels and titles
    plt.title('RAG Pipeline Evaluation Scores', fontsize=16)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(np.concatenate((x_ir, x_ragas)), all_metrics, rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add the value on top of each bar
    for bars in [bars1, bars2]:
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    
    # Save the figure instead of showing it
    plt.savefig(filename)
    print(f"Chart saved to {filename}")
    plt.close() # Close the figure to free up memory


def main():
    logger.info("Pipeline started")
    start = time.time()

    # Step 1: Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding_dim = model.get_sentence_embedding_dimension()

    # Step 2: Choose dataset and load it
    dataset_name = "squad"
    subset_name = "validation"  # We will use this to name the files
    
    # You can change this parameter to evaluate more or fewer records.
    num_records_to_evaluate = 10
    
    dataframes = get_dataset(dataset_name=dataset_name, data_dir="data/input")

    train_df = dataframes.get("train")
    val_df = dataframes.get("validation")
    
    logger.info(f"Train split: {None if train_df is None else train_df.shape}")
    logger.info(f"Validation split: {None if val_df is None else val_df.shape}")

    if train_df is not None:
        logger.info(f"Sample rows:\n{train_df.head().to_string()}")

    # Step 3: Define paths for cached index and documents in the new directory
    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, f"{dataset_name}_{subset_name}_index.bin")
    doc_path = os.path.join(output_dir, f"{dataset_name}_{subset_name}_docs.json")

    # Step 4: Init and load/build Retriever
    logger.info("Loading or building FAISS index...")
    
    # New logic: call the static method to handle loading or building
    retriever = FAISSRetriever.load_or_build(
        index_path=index_path, 
        doc_path=doc_path, 
        embedding_dim=embedding_dim, 
        embed_fn=model.encode, 
        documents_to_add=prepare_documents(val_df, dataset_name)
    )

    logger.info("Retriever is ready")
    
    end = time.time()
    logger.info(f"Total startup time: {end - start:.2f} sec")

    # Step 5: Init RAG pipeline
    rag = NaiveRAG(retriever)
    
    # Step 6: Evaluate on a subset of records
    logger.info(f"Starting evaluation on {num_records_to_evaluate} validation records...")
    
    ir_eval = IREvaluator()
    ragas_embed_eval = WorkingRagasEmbeddingEvaluator(model) # Use our new, working evaluator
    
    ir_scores_list = []
    ragas_embed_scores_list = []

    # Loop through the specified number of records in the validation DataFrame
    for index, row in val_df.head(num_records_to_evaluate).iterrows():
        # Step 7: Get query and ground truth for the current record
        query = row["question"]
        ground_truth = extract_ground_truth(row, dataset_name)
        
        # Step 8: Run the RAG pipeline for the current record
        # We manually run the retriever and provide a placeholder answer
        # since we don't have a real LLM connected to generate one.
        retrieved_docs = retriever.retrieve(query, top_k=5)
        answer = "This is a placeholder answer." # This is required for Ragas evaluation

        # Step 9: Evaluate the RAG pipeline for the current record
        ir_scores = ir_eval.evaluate(query, retrieved_docs, ground_truth)
        ragas_embed_scores = ragas_embed_eval.evaluate(query, answer, retrieved_docs, ground_truth)

        ir_scores_list.append(ir_scores)
        ragas_embed_scores_list.append(ragas_embed_scores)

    # Step 10: Calculate average scores after the loop finishes
    if ir_scores_list:
        avg_ir_scores = {k: np.mean([d[k] for d in ir_scores_list if k in d]) for k in ir_scores_list[0]}
    else:
        avg_ir_scores = {}
        
    if ragas_embed_scores_list:
        avg_ragas_embed_scores = {k: np.mean([d[k] for d in ragas_embed_scores_list if k in d]) for k in ragas_embed_scores_list[0]}
    else:
        avg_ragas_embed_scores = {}

    print("\n📊 Average IR Scores:", avg_ir_scores)
    print("\n📊 Average RAGAS Embedding Scores:", avg_ragas_embed_scores)

    # Step 11: Visualize and save the average scores
    
    logger.info(f"Combined scores for visualization: IR:{avg_ir_scores} | RAGAS:{avg_ragas_embed_scores}")
    
    chunk_overlap = 0
    results_dir = "data/results"
    os.makedirs(results_dir, exist_ok=True)
    filename = os.path.join(results_dir, f"{dataset_name}_{subset_name}_{chunk_overlap}_results.png")

    if avg_ir_scores or avg_ragas_embed_scores:
        visualize_scores(avg_ir_scores, avg_ragas_embed_scores, filename)
    else:
        logger.warning("No scores to visualize. No chart was generated.")

if __name__ == "__main__":
    main()
