from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

class Chunker:
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 120):
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
    def chunk(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)
    
chunker = Chunker()