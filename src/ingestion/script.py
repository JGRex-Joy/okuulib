from src.ingestion.load_pdf import PDFLoader
from src.ingestion.chunk import chunker
from src.shared.embedder import embedder
from src.shared.vector_store import vector_store

def main():
    pdf_loader = PDFLoader("data\er-toshtuk.pdf")
    documents = pdf_loader.load_pdf()
    print(f'Loaded {len(documents)} docs')
    
    chunks = chunker.chunk(documents)
    print(f'Chunked {len(chunks)} docs')
    
    texts = [doc.page_content for doc in chunks]
    embeddings = embedder.embed_batch(texts)
    print(f'Embeddings genereted for {len(embeddings)} chunks')
    
    ids = list(range(len(embeddings)))
    payloads = [{"text": text} for text in texts]
    vector_store.add(ids, embeddings, payloads)
    
    vector_store.add(embeddings)
    print("Saved fo Qdrant")
    
if __name__ == "__main__":
    main()