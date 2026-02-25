from fastapi import FastAPI
from pydantic import BaseModel

from src.ingestion.api.router import router as ingestion_router
from src.retrieval.api.router import router as retrieval_router

app = FastAPI(
    title="OkuuLib API",
    description="RAG-сервис для кыргызской литературы",
    version="1.0.0",
)

# Routers
app.include_router(ingestion_router)
app.include_router(retrieval_router)


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    message: str


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(status="ok", message="OkuuLib is running")