from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from typing import List

class PDFLoader:
    def __init__(self, path: str):
        self.path = path
        
    def load_pdf(self) -> List[Document]:
        loader = PyPDFLoader(self.path)
        return loader.lazy_load()
