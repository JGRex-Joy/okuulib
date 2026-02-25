from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, SparseVector
)
from typing import List, Dict

from src.config import settings


class VectorStore:
    def __init__(self):
        self.collection_name = settings.COLLECTION_NAME
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=settings.QDRANT_TIMEOUT,
        )
        self._ensure_collection(vector_size=settings.VECTOR_SIZE)

    def _ensure_collection(self, vector_size: int):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams()
                },
            )

            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="book",
                field_schema="keyword",
            )

            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="text",
                field_schema="text",
            )

    def add(
        self,
        ids: List[str],
        vectors: List[List[float]],
        sparse_vectors: List[SparseVector],
        payloads: List[Dict],
        batch_size: int = settings.QDRANT_BATCH_SIZE,
    ):
        total = len(ids)
        for i in range(0, total, batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_vectors = vectors[i:i + batch_size]
            batch_sparse = sparse_vectors[i:i + batch_size]
            batch_payloads = payloads[i:i + batch_size]

            points = [
                PointStruct(
                    id=batch_ids[j],
                    vector={
                        "dense": batch_vectors[j],
                        "sparse": batch_sparse[j],
                    },
                    payload=batch_payloads[j],
                )
                for j in range(len(batch_ids))
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            print(f"Uploaded {(i + len(batch_ids)) / total * 100:.1f}%")

    def delete_by_book(self, book_name: str) -> int:
        count_before = self.client.count(
            collection_name=self.collection_name,
            count_filter=Filter(
                must=[FieldCondition(key="book", match=MatchValue(value=book_name))]
            ),
        ).count

        if count_before == 0:
            return 0

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                                key="book", match=MatchValue(value=book_name)
                            )
                        ]
                )
            ),
        )

        return count_before


vector_store = VectorStore()