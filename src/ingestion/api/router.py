import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from src.ingestion.api.schemas import IngestResponse, DeleteResponse
from src.ingestion.ingest import ingest_book
from src.shared.qdrant.vector_store import vector_store

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


@router.post("/upload", response_model=IngestResponse)
async def upload_and_ingest(file: UploadFile = File(...)):
    if not file.filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="Only .docx files are supported")

    save_path = DATA_DIR / file.filename
    with save_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    success = ingest_book(save_path)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to ingest '{file.filename}'")

    return IngestResponse(
        book_name=save_path.stem,
        message=f"Book '{save_path.stem}' successfully ingested.",
    )


@router.delete("/delete/{book_name}", response_model=DeleteResponse)
async def delete_book(book_name: str):

    deleted = vector_store.delete_by_book(book_name)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"No chunks found for book '{book_name}'")

    file_path = DATA_DIR / f"{book_name}.docx"
    if file_path.exists():
        file_path.unlink()

    return DeleteResponse(
        book_name=book_name,
        deleted_chunks=deleted,
        message=f"Deleted {deleted} chunks for book '{book_name}'.",
    )