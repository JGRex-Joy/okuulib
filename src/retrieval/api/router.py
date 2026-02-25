from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.retrieval.services.rag_service import rag_service
from src.retrieval.api.schemas import AskRequest, AskResponse

router = APIRouter(prefix="/retrieval", tags=["Retrieval"])


@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if not request.book_name.strip():
        raise HTTPException(status_code=400, detail="book_name cannot be empty")

    answer = await rag_service.ask(request.query, request.book_name)

    return AskResponse(
        answer=answer,
        book_name=request.book_name,
        query=request.query,
    )