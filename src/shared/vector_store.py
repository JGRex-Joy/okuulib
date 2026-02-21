from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, SparseVector
from typing import List, Dict, Optional

import os
from dotenv import load_dotenv
load_dotenv()

class VectorStore:
    def __init__(self):
        self.collection_name = "okuulib"
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=60
        )
        
        self._ensure_collection(vector_size=1536)
        
    def _ensure_collection(self, vector_size: int):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams()
                }
            )
            
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="book",
                field_schema="keyword"
            )
            
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="text",
                field_schema="text"
            )

    def add(
        self,
        ids: List[str],
        vectors: List[List[float]],
        sparse_vectors: List[SparseVector],
        payloads: List[Dict],
        batch_size: int = 16
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
                        "sparse": batch_sparse[j]   
                    },
                    payload=batch_payloads[j]
                )
                for j in range(len(batch_ids))
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            print(f'Uploaded {(i + len(batch_ids)) / total * 100:.1f}%')
    
            
    def search(
        self,
        query_vector: List[float],
        query_sparse: SparseVector,
        query_text: str,
        top_k: int = 5,
        book_name: Optional[str] = None
    ):
        query_filter: Optional[Filter] = None
        
        if book_name:
            query_filter = Filter(
                must = [
                    FieldCondition(
                        key = "book",
                        match = MatchValue(value=book_name)
                    )
                ]
            )
        
        return self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                # Dense - семантика
                models.Prefetch(
                    query=query_vector,
                    using="default",
                    limit=top_k*3,
                    filter=query_filter
                ),
                # Sparse - bm25
                models.Prefetch(
                    query=query_sparse,
                    using="sparse",
                    limit=top_k*3,
                    filter=query_filter
                )
            ],
            
            query=models.FusionQuery(
                fusion=models.Fusion.RRF
            ),
            limit=top_k
        )
        
vector_store = VectorStore()