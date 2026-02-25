from fastapi import FastAPI

from src.health import router as health_router
from src.ingestion.api.router import router as ingestion_router
from src.retrieval.api.router import router as retrieval_router

app = FastAPI(
    title="OkuuLib API",
    description="RAG-сервис для кыргызской литературы",
    version="1.0.0",
)

app.include_router(health_router)
app.include_router(ingestion_router)
app.include_router(retrieval_router)