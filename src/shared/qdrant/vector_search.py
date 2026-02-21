from qdrant_client import models
from qdrant_client.models import Filter, FieldCondition, MatchValue, SparseVector
from typing import List, Optional

from src.shared.qdrant.vector_store import vector_store
from src.config import settings

class VectorSearch:
    def __init__(self):
        self.client = vector_store.client
        self.collection_name = vector_store.collection_name
        
    def search(
        self,
        query_vector: List[float],
        query_sparse: SparseVector,
        book_name: Optional[str] = None
    ):
        
        query_filter: Optional[Filter] = None
        
        if book_name:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="book",
                        match=MatchValue(value=book_name)
                    )
                ]
            )
            
        return self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                # Dense - семантика
                models.Prefetch(
                    query=query_vector,
                    using="dense",
                    limit = settings.TOP_K * settings.PREFETCH_MULTIPLIER,
                    filter = query_filter
                ),
                
                # Sparse - текстовка
                models.Prefetch(
                    query=query_sparse,
                    using="sparse",
                    limit = settings.TOP_K * settings.PREFETCH_MULTIPLIER,
                    filter=query_filter
                )
            ],
            query=models.FusionQuery(
                fusion=models.Fusion.RRF
            ),
            limit=settings.TOP_K
        )
        
vector_search = VectorSearch()