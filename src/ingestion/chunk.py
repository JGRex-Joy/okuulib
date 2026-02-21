from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

from src.config import settings

class Chunker:
    def __init__(self):
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
    def chunk(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)
    
chunker = Chunker()