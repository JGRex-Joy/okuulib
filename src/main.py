from fastapi import FastAPI

from src.retrieval.services.rag_service import rag_service
from src.models import AskRequest, AskResponse

app = FastAPI()

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    answer = await rag_service.ask(request.query, request.book_name)
    return AskResponse(answer=answer)