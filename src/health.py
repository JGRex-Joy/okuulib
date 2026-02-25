from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["System"])


class HealthResponse(BaseModel):
    status: str
    message: str


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", message="OkuuLib is running")