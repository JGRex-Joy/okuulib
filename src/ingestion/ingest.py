from src.ingestion.load_docx import DocxLoader
from src.ingestion.chunk import chunker
from src.shared.embedders.dense_embedder import dense_embedder
from src.shared.embedders.sparse_embedder import sparse_embedder
from src.shared.qdrant.vector_store import vector_store

from pathlib import Path
import uuid
import time

DATA_DIR = Path("data")


def ingest_book(book_path: Path, retries: int = 3) -> bool:
    print(f"\n{'='*50}")
    print(f"📖 Book: {book_path.stem}")
    print(f"{'='*50}")

    loader = DocxLoader(path=str(book_path))
    documents = list(loader.load_docx())
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
            "chunk_id": i,
        }
        for i, text in enumerate(texts)
    ]

    for attempt in range(1, retries + 1):
        try:
            vector_store.add(ids, dense_vectors, sparse_vectors, payloads)
            print(f"✅ Saved to Qdrant: {book_path.stem}")
            return True
        except Exception as e:
            print(f"⚠️ Retry {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(5)

    print(f"❌ Could not save: {book_path.stem}")
    return False


def main():
    books = list(DATA_DIR.glob("*.docx"))

    if not books:
        print("❌ No .docx books found in data/")
        return

    print(f"\n🚀 Books found: {len(books)}")
    for book in books:
        print(f"  - {book.stem}")

    failed = []
    for i, book_path in enumerate(books, 1):
        print(f"\n[{i}/{len(books)}]", end="")
        success = ingest_book(book_path)
        if not success:
            failed.append(book_path.stem)

    print(f"\n{'='*50}")
    print(f"🎉 Ready! Loaded: {len(books) - len(failed)}/{len(books)}")
    if failed:
        print(f"❌ Could not load: {', '.join(failed)}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()