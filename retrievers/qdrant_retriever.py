from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
import numpy as np

class QdrantRetriever:
    def __init__(self, url: str, collection: str, dim: int):
        self.client = QdrantClient(url=url)
        self.collection = collection
        self.dim = dim

    def recreate(self):
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE)
        )

    def upsert(self, ids, vectors, payloads):
        self.client.upsert(self.collection, points=[qm.PointStruct(id=i, vector=v.tolist(), payload=p) for i,v,p in zip(ids, vectors, payloads)])

    def search(self, vector: np.ndarray, top_k: int, filters: dict | None = None):
        cond = None
        if filters:
            must = [qm.FieldCondition(key=k, match=qm.MatchValue(value=v)) for k,v in filters.items()]
            cond = qm.Filter(must=must)
        res = self.client.search(collection_name=self.collection, query_vector=vector.ravel().tolist(), limit=top_k, query_filter=cond)
        return [(str(p.id), float(p.score)) for p in res]
