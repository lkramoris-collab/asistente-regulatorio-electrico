import os
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DOCUMENTS_DIR = "documents"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "reglamento"
CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 200  # overlap between chunks


def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += f"\n[Página {page_num + 1}]\n{page_text}"
    return text


def chunk_text(text, source_name, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at a sentence boundary
        if end < len(text):
            last_period = chunk.rfind(".")
            last_newline = chunk.rfind("\n")
            break_point = max(last_period, last_newline)
            if break_point > chunk_size * 0.5:
                end = start + break_point + 1
                chunk = text[start:end]

        chunks.append({
            "text": chunk.strip(),
            "source": source_name,
            "chunk_index": len(chunks)
        })

        start = end - overlap

    return chunks


def main():
    print("=" * 60)
    print("INGESTION: Loading documents into vector database")
    print("=" * 60)

    # Initialize embedding model
    print("\n[1/4] Loading embedding model (first time takes a minute)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Initialize ChromaDB
    print("[2/4] Setting up ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if it exists (fresh start)
    try:
        client.delete_collection(COLLECTION_NAME)
        print("  - Cleared existing collection")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # Process all PDFs
    print(f"[3/4] Reading documents from '{DOCUMENTS_DIR}/'...")
    all_chunks = []

    # Process PDFs in main folder
    for filename in os.listdir(DOCUMENTS_DIR):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(DOCUMENTS_DIR, filename)
            print(f"  - Processing PDF: {filename}")

            text = extract_text_from_pdf(filepath)
            chunks = chunk_text(text, filename)
            all_chunks.extend(chunks)

            print(f"    -> {len(chunks)} chunks created ({len(text)} characters)")

    # Process scraped .txt files
    scraped_dir = os.path.join(DOCUMENTS_DIR, "scraped")
    if os.path.exists(scraped_dir):
        for filename in os.listdir(scraped_dir):
            if filename.lower().endswith(".txt") and not filename.startswith("_"):
                filepath = os.path.join(scraped_dir, filename)
                print(f"  - Processing TXT: scraped/{filename}")

                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()

                chunks = chunk_text(text, f"web:{filename}")
                all_chunks.extend(chunks)

                print(f"    -> {len(chunks)} chunks created ({len(text)} characters)")

    if not all_chunks:
        print("ERROR: No documents found!")
        return

    # Generate embeddings and store in ChromaDB
    print(f"[4/4] Generating embeddings for {len(all_chunks)} chunks...")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(all_chunks))],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[{"source": c["source"], "chunk_index": c["chunk_index"]} for c in all_chunks]
    )

    print(f"\nDONE! {len(all_chunks)} chunks stored in ChromaDB.")
    print(f"Sources processed:")
    sources = set(c["source"] for c in all_chunks)
    for s in sources:
        count = sum(1 for c in all_chunks if c["source"] == s)
        print(f"  - {s}: {count} chunks")


if __name__ == "__main__":
    main()
