import re
from typing import List
from langchain_core.documents import Document

class PdfCleaner:
    def __init__(self):
        self.patterns = [
            r"Bizdin.kg",
            r"bizdin.kg",
            r"www.bizdin.kg",
            r"Bizdin\.kg",
            r"www\.bizdin\.kg"
        ]
        
    def clean_text(self, text: str) -> str:
        for pattern in self.patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
            
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def clean_documents(self, documents: List[Document]) -> List[Document]:
        for doc in documents:
            doc.page_content = self.clean_text(doc.page_content)
        return documents
    
pdfCleaner = PdfCleaner()
