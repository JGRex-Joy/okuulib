from shared.embedders.dense_embedder import dense_embedder
from shared.embedders.sparse_embedder import sparse_embedder
from src.shared.vector_store import vector_store
from src.retrieval.services.llm_service import llm_service

class RAGService:
    
    async def ask(self, query: str, book_name: str) -> str:
        dense_vector = dense_embedder.embed(query)
        sparse_vector = sparse_embedder.embed(query)
        
        found = vector_store.search(
            query_vector=dense_vector,
            query_sparse=sparse_vector,
            book_name=book_name
        )
        
        contexts = [
            point.payload["text"]
            for point in found.points
        ]
        
        return llm_service.generate_answer(query, contexts)
    
rag_service = RAGService()