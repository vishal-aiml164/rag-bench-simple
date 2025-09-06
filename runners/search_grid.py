import argparse, yaml, random
from utils.logger import log
from retrievers.hybrid_retriever import Doc, HybridRetriever
from evaluators.traditional_metrics import recall_at_k, mrr_at_k, ndcg_at_k

def load_dataset(name: str):
    docs = [
        Doc("d1", "Aspirin is an analgesic and antipyretic.", {"class": "NSAID"}),
        Doc("d2", "Ibuprofen reduces inflammation and pain.", {"class": "NSAID"}),
        Doc("d3", "Metformin improves insulin sensitivity.", {"class": "biguanide"}),
    ]
    queries = [
        {"id": "q1", "text": "pain relief nsaid", "relevant": ["d1","d2"], "filters": {"class": "NSAID"}},
        {"id": "q2", "text": "improve insulin sensitivity", "relevant": ["d3"], "filters": {}},
    ]
    return docs, queries

def score(r, queries, k=20):
    s = 0.0
    for q in queries:
        ranked = [cid for cid,_ in r.search(q["text"], top_k=k, filters=q.get("filters", {}))]
        s += ndcg_at_k(ranked, q["relevant"], k)
    return s / len(queries)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default="config/retriever_config.yaml")
    ap.add_argument("--method", choices=["grid","random"], default="random")
    ap.add_argument("--trials", type=int, default=10)
    args = ap.parse_args()

    # toy dataset
    docs, queries = load_dataset("toy")

    # search space (keep it simple, override from YAML if desired)
    alphas = [0.2, 0.5, 0.8]
    topks = [10, 20, 50]

    best, best_score = None, -1
    for _ in range(args.trials):
        a = random.choice(alphas)
        k = random.choice(topks)
        r = HybridRetriever(docs, alpha=a)
        s = score(r, queries, k=k)
        if s > best_score:
            best, best_score = {"alpha": a, "top_k": k}, s
        log(f"trial -> alpha={a} top_k={k} ndcg@k={s:.3f}")

    log(f"BEST => {best} score={best_score:.3f}")

if __name__ == "__main__":
    main()
