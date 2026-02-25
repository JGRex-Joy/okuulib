from pydantic import BaseModel

class DeleteResponse(BaseModel):
    book_name: str
    deleted_chunks: int
    message: str

class IngestResponse(BaseModel):
    book_name: str
    message: str