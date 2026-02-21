from src.ingestion.load_pdf import PDFLoader
from src.ingestion.chunk import chunker
from src.shared.embedders.dense_embedder import dense_embedder
from src.shared.embedders.sparse_embedder import sparse_embedder
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
    
    dense_vectors = dense_embedder.embed_batch(texts)
    print(f'Dense embeddings genereted: {len(dense_vectors)}')
    
    sparse_vectors = sparse_embedder.embed_batch(texts)
    print(f'Sparse embeddings generated: {len(sparse_vectors)}')
    
    ids = [str(uuid.uuid4()) for _ in texts]
    payloads = [
        {
        "text": text,
        "book": Path(book_path).stem,
        "chunk_id": i
        }
        for i, text in enumerate(texts)
    ]
    vector_store.add(ids, dense_vectors, sparse_vectors, payloads)
    
    print("Saved fo Qdrant")
    
if __name__ == "__main__":
    main()