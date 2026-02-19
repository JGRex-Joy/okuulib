from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
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
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )

    def add(
        self,
        ids: List[int],
        vectors: List[List[float]],
        payloads: List[Dict],
        batch_size: int = 16
    ):
        total = len(ids)
        for i in range(0, total, batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_vectors = vectors[i:i + batch_size]
            batch_payloads = payloads[i:i + batch_size]

            points = [
                PointStruct(
                    id=batch_ids[j],
                    vector=batch_vectors[j],
                    payload=batch_payloads[j],
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
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ):
        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=filters,
        )
        
vector_store = VectorStore()