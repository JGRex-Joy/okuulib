from fastapi import FastAPI
from src.retrieval.services.rag_service import rag_service

app = FastAPI()

@app.post("/ask")
async def ask(query: str, book_name: str):
    return await rag_service.ask(query, book_name)