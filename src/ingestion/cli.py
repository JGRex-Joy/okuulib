"""
Консольный запуск индексации всех книг из /data.

Использование:
  Все книги:       python -m src.ingestion.cli
  Одна книга:      python -m src.ingestion.cli data/книга.docx
"""

import sys
import argparse
from pathlib import Path

from src.ingestion.ingest import ingest_book

DATA_DIR = Path("data")


def run(target: Path, clean: bool) -> None:
    if target.is_file():
        books = [target]
    else:
        books = list(target.glob("*.docx"))

    if not books:
        print(f"❌ No .docx files in {target}")
        sys.exit(1)

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


def main():
    parser = argparse.ArgumentParser(description="Indexing to Qdrant")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DATA_DIR),
        help="Path to .docx files or folder (data/)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean file",
    )
    args = parser.parse_args()

    target = Path(args.path)
    if not target.exists():
        print(f"❌ Path does not exist: {target}")
        sys.exit(1)

    run(target, clean=args.clean)


if __name__ == "__main__":
    main()