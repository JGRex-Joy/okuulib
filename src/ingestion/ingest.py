from src.ingestion.load_pdf import PDFLoader
from src.ingestion.chunk import chunker
from src.shared.embedders.dense_embedder import dense_embedder
from src.shared.embedders.sparse_embedder import sparse_embedder
from src.shared.qdrant.vector_store import vector_store
from src.ingestion.clean import pdfCleaner

from pathlib import Path
import uuid

DATA_DIR = Path("data")

def ingest_book(book_path: Path):
    print(f"\n{'='*50}")
    print(f"📖 Book: {book_path.stem}")
    print(f"{'='*50}")

    pdf_loader = PDFLoader(path=str(book_path))
    documents = list(pdf_loader.load_pdf())
    documents = pdfCleaner.clean_documents(documents)
    print(f"✅ Pages loaded: {len(documents)}")

    chunks = chunker.chunk(documents)
    print(f"✅ Chunks: {len(chunks)}")

    texts = [doc.page_content for doc in chunks]

    dense_vectors = dense_embedder.embed_batch(texts)
    print(f"✅ Dense embeddings: {len(dense_vectors)}")

    sparse_vectors = sparse_embedder.embed_batch(texts)
    print(f"✅ Sparse embeddings: {len(sparse_vectors)}")

    ids = [str(uuid.uuid4()) for _ in texts]
    payloads = [
        {
            "text": text,
            "book": book_path.stem,
            "chunk_id": i
        }
        for i, text in enumerate(texts)
    ]

    vector_store.add(ids, dense_vectors, sparse_vectors, payloads)
    print(f"✅ Saved to Qdrant: {book_path.stem}")


def main():
    books = list(DATA_DIR.glob("*.pdf"))

    if not books:
        print("❌ PDF not found in data/")
        return

    print(f"\n🚀 Books found: {len(books)}")
    for book in books:
        print(f"  - {book.stem}")

    for i, book_path in enumerate(books, 1):
        print(f"\n[{i}/{len(books)}]", end="")
        ingest_book(book_path)

    print(f"\n{'='*50}")
    print(f"🎉 Ready! Books loaded: {len(books)}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()