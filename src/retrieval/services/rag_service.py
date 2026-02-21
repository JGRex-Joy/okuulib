from src.shared.embedder import embedder
from src.shared.vector_store import vector_store
from src.retrieval.services.llm_service import llm_service

class RAGService:
    
    async def ask(self, query: str, book_name: str) -> str:
        embed_query = embedder.embed(query)
        
        found = vector_store.search(
            query_vector=embed_query,
            query_text=query,
            book_name=book_name
        )
        
        contexts = [
            point.payload["text"]
            for point in found.points
        ]
        
        return llm_service.generate_answer(query, contexts)
    
rag_service = RAGService()