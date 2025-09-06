import matplotlib.pyplot as plt

def visualize_scores(ir_scores, ragas_scores, filename="results.png"):
    """Visualize IR and Ragas embedding scores side by side as bar charts."""
    plt.figure(figsize=(10, 6))

    # IR scores
    if ir_scores:
        plt.bar(range(len(ir_scores)), list(ir_scores.values()), alpha=0.7, label="IR Scores")
    # Ragas embedding scores
    if ragas_scores:
        plt.bar(
            [i + 0.4 for i in range(len(ragas_scores))],
            list(ragas_scores.values()),
            alpha=0.7,
            label="Ragas Embedding Scores"
        )

    # X-ticks
    keys = list(ir_scores.keys()) + list(ragas_scores.keys())
    plt.xticks(range(len(keys)), keys, rotation=45, ha="right")

    plt.ylabel("Score")
    plt.title("Evaluation Scores Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
