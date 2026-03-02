from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.health import router as health_router
from src.ingestion.api.router import router as ingestion_router
from src.retrieval.api.router import router as retrieval_router

app = FastAPI(
    title="OkuuLib API",
    description="RAG-сервис для кыргызской литературы",
    version="1.0.0",
)

ALLOWED_ORIGINS = [
    "*",                          
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(health_router)
app.include_router(ingestion_router)
app.include_router(retrieval_router)