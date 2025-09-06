from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

@dataclass
class Doc:
    doc_id: str
    text: str
    metadata: Dict[str, Any]

class EmbeddingProvider:
    def __init__(self, model_name: str = "al-mini-l6"):
        from sklearn.feature_extraction.text import HashingVectorizer
        self._hash = HashingVectorizer(n_features=2**12, alternate_sign=False, norm=None)
    def embed(self, texts: List[str]):
        X = self._hash.transform(texts)
        norms = np.sqrt(X.power(2).sum(axis=1))
        norms[norms == 0] = 1.0
        X = X.multiply(1.0 / norms)
        return X

class SparseRetriever:
    def __init__(self):
        self.vec = TfidfVectorizer(ngram_range=(1,2), max_df=0.95)
        self.doc_term = None
        self.ids: List[str] = []
    def fit(self, ids: List[str], texts: List[str]):
        self.ids = ids
        self.doc_term = self.vec.fit_transform(texts)
        return self
    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        q = self.vec.transform([query])
        scores = (self.doc_term @ q.T).toarray().ravel()
        idx = np.argsort(-scores)[:top_k]
        return [(self.ids[i], float(scores[i])) for i in idx]

class DenseRetriever:
    def __init__(self, emb: EmbeddingProvider):
        self.emb = emb
        self.nn = None
        self.mat = None
        self.ids: List[str] = []
    def fit(self, ids: List[str], texts: List[str]):
        self.ids = ids
        self.mat = self.emb.embed(texts)
        self.nn = NearestNeighbors(metric="cosine").fit(self.mat)
        return self
    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        qv = self.emb.embed([query])
        d, idx = self.nn.kneighbors(qv, n_neighbors=top_k)
        sims = 1.0 - d.ravel()
        idx = idx.ravel()
        return [(self.ids[i], float(sims[j])) for j, i in enumerate(idx)]

class HybridRetriever:
    def __init__(self, docs: List[Doc], alpha: float = 0.5, embedder: Optional[EmbeddingProvider] = None):
        self.docs = {d.doc_id: d for d in docs}
        self.alpha = alpha
        self.embedder = embedder or EmbeddingProvider()
        self.sparse = SparseRetriever()
        self.dense = DenseRetriever(self.embedder)
        ids = [d.doc_id for d in docs]
        texts = [d.text for d in docs]
        self.sparse.fit(ids, texts)
        self.dense.fit(ids, texts)

    def _filter(self, ids: List[str], filters: Dict[str, Any]) -> List[str]:
        if not filters: return ids
        out = []
        for cid in ids:
            md = self.docs[cid].metadata
            ok = True
            for k, v in filters.items():
                if k not in md: ok=False; break
                if isinstance(md[k], (list,tuple,set)):
                    if v not in md[k]: ok=False; break
                else:
                    if md[k] != v: ok=False; break
            if ok: out.append(cid)
        return out

    def search(self, query: str, top_k: int, filters: Optional[Dict[str, Any]] = None):
        s_hits = self.sparse.search(query, top_k=max(top_k, 50))
        d_hits = self.dense.search(query, top_k=max(top_k, 50))

        def norm(hits):
            if not hits: return {}
            scores = np.array([h[1] for h in hits]).reshape(-1,1)
            ns = MinMaxScaler().fit_transform(scores).ravel()
            return {hits[i][0]: float(ns[i]) for i in range(len(hits))}

        sm = norm(s_hits)
        dm = norm(d_hits)
        all_ids = set(sm) | set(dm)
        merged = []
        for cid in all_ids:
            score = self.alpha * sm.get(cid, 0.0) + (1-self.alpha) * dm.get(cid, 0.0)
            merged.append((cid, score))
        merged.sort(key=lambda x: -x[1])
        ranked = [cid for cid,_ in merged]
        ranked = self._filter(ranked, filters or {})
        return [(cid, dict(merged)[cid]) for cid in ranked[:top_k]]
