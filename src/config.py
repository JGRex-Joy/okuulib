from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    
    # Qdrant
    QDRANT_URL: str
    QDRANT_API_KEY: str
    QDRANT_TIMEOUT: int = 60
    COLLECTION_NAME: str = "okuulib"
    QDRANT_BATCH_SIZE: int = 16
    
    # Vector size
    VECTOR_SIZE: int = 1536
    
    # OpenAI
    OPENAI_API_KEY: str
    
    # Dense embedder
    DENSE_EMBEDDING_MODEL: str = "text-embedding-3-small"
    DENSE_EMBEDDING_BATCH_SIZE: int = 100
    
    # Sparse embedder
    SPARSE_EMBEDDING_MODEL: str = "Qdrant/bm25"
    
    # LLM
    LLM_MODEL: str = "gpt-4.1-mini"
    
    # Search
    TOP_K: int = 10
    PREFETCH_MULTIPLIER: int = 3
    
    # Chunking
    CHUNK_SIZE: int = 700
    CHUNK_OVERLAP: int = 120
    
    class Config:
        env_file = ".env"
        
settings = Settings()
