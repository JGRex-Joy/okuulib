from src.shared.embedders.dense_embedder import dense_embedder
from src.shared.embedders.sparse_embedder import sparse_embedder
from src.shared.qdrant.vector_search import vector_search
from src.retrieval.services.llm_service import llm_service

class RAGService:
    
    async def ask(self, query: str, book_name: str) -> str:
        dense_vector = dense_embedder.embed(query)
        sparse_vector = sparse_embedder.embed(query)
        
        found = vector_search.search(
            query_vector=dense_vector,
            query_sparse=sparse_vector,
            book_name=book_name
        )
        
        print(f"\n🔍 Найдено чанков: {len(found.points)}")
        for p in found.points:
            print(f"  book: {p.payload['book']} | score: {p.score:.3f}")
            print(f"  text: {p.payload['text'][:100]}")
            print()
            
        contexts = [
            point.payload["text"]
            for point in found.points
        ]
        
        return llm_service.generate_answer(query, contexts)
    
rag_service = RAGService()