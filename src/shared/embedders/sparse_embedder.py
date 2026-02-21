from fastembed import SparseTextEmbedding
from qdrant_client import models
from typing import List

class SparseEmbedder:
    def __init__(self):
        self.model = SparseTextEmbedding(model_name="Qdrant/bm25")
        
    def embed(self, text: str) -> models.SparseVector:
        result = list(self.model.embed([text]))[0]
        return models.SparseVector(
            indices = result.indices.tolist(),
            values = result.values.tolist()
        )
        
    def embed_batch(self, texts: List[str]) -> List[models.SparseVector]:
        results = list(self.model.embed(texts))
        return [
            models.SparseVector(
                indices = r.indices.tolist(),
                values = r.values.tolist()
            )
            for r in results
        ]
        
sparse_embedder = SparseEmbedder()