from pydantic import BaseModel

class AskRequest(BaseModel):
    query: str
    book_name: str
    
class AskResponse(BaseModel):
    answer: str