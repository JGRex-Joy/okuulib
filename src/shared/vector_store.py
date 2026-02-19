from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Optional

import os
from dotenv import load_dotenv
load_dotenv()

class VectorStore:
    def __init__(self):
        self.collection_name = "okuulib",
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        
        self._ensure_collection(vector_size=1536)
        
    def _ensure_collection(self, vector_size: int):
        collections = self.client.get_collections().collections
        existing = [c.name for c in collections]
        
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                ),
            )
            
    def add(
        self,
        ids: List[int],
        vectors: List[List[float]],
        payloads: List[Dict],
    ):
        points = [
            PointStruct(
                id=ids[i],
                vector=vectors[i],
                payload=payloads[i],
            )
            for i in range(len(ids))
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,        
        )
        
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