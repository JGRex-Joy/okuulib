from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document
from typing import List


class DocxLoader:
    def __init__(self, path: str):
        self.path = path

    def load_docx(self) -> List[Document]:
        loader = Docx2txtLoader(self.path)
        return loader.lazy_load()
    