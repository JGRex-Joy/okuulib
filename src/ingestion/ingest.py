from src.ingestion.load_pdf import PDFLoader
from src.ingestion.chunk import chunker
from src.shared.embedder import embedder
from src.shared.vector_store import vector_store
from src.ingestion.clean import pdfCleaner

from pathlib import Path
import uuid

book_path: str = "data\janyl-myrza.pdf"
    
def main():
    pdf_loader = PDFLoader(path=book_path)
    documents = list(pdf_loader.load_pdf())
    documents = pdfCleaner.clean_documents(documents)
    
    print(f'Loaded {len(documents)} docs')
    
    chunks = chunker.chunk(documents)
    print(f'Chunked {len(chunks)} docs')
    
    texts = [doc.page_content for doc in chunks]
    embeddings = embedder.embed_batch(texts)
    print(f'Embeddings genereted for {len(embeddings)} chunks')
    
    ids = [str(uuid.uuid4()) for _ in embeddings]
    payloads = [
        {
        "text": text,
        "book": Path(book_path).stem,
        "chunk_id": i
        }
        for i, text in enumerate(texts)
    ]
    vector_store.add(ids, embeddings, payloads)
    
    print("Saved fo Qdrant")
    
if __name__ == "__main__":
    main()