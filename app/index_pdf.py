import pdfplumber
from openai import OpenAI
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType
)
import hashlib
import re

# -------------------------------
# Configuration
# -------------------------------
COLLECTION_NAME = "books_rag"

# Connect to Milvus
connections.connect(alias="default", host="localhost", port="19530")

# Initialize OpenAI
openai_client = OpenAI()


# -------------------------------
# PDF Utilities
# -------------------------------
def read_pdf(pdf_path: str) -> str:
    """Extract text from PDF using pdfplumber."""
    all_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text.append(text.strip())
    return "\n".join(all_text)


def semantic_chunk_text(text: str, max_chunk_size: int = 800, overlap: int = 100):
    """
    Split text into semantic chunks (by paragraphs, then merge if too small).
    """
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) + 1 <= max_chunk_size:
            current_chunk += (" " if current_chunk else "") + para
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para
    if current_chunk:
        chunks.append(current_chunk.strip())

    # Add overlap
    if overlap > 0 and len(chunks) > 1:
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                prev_tail = chunks[i-1][-overlap:]
                overlapped_chunks.append(prev_tail + " " + chunk)
        chunks = overlapped_chunks

    return chunks


def split_long_chunk(text: str, max_len: int = 2000):
    """
    Split a long chunk into smaller ones on sentence boundaries.
    """
    if len(text) <= max_len:
        return [text]

    sentences = re.split(r'(?<=[.!?])\s+', text)
    sub_chunks, current = [], ""

    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_len:
            current += (" " if current else "") + sent
        else:
            if current:
                sub_chunks.append(current.strip())
            current = sent
    if current:
        sub_chunks.append(current.strip())

    return sub_chunks


def enforce_safe_chunks(chunks, max_len=2000):
    safe = []
    for c in chunks:
        safe.extend(split_long_chunk(c, max_len=max_len))
    return safe


# -------------------------------
# OpenAI Embeddings
# -------------------------------
def embed_text(text: str):
    """Generate embedding using OpenAI."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def chunk_hash(text: str) -> str:
    """Generate SHA256 hash for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# -------------------------------
# Index PDF into Milvus
# -------------------------------
def index_pdf(pdf_path: str, source: str):
    text = read_pdf(pdf_path)
    chunks = semantic_chunk_text(text, max_chunk_size=800, overlap=100)
    chunks = enforce_safe_chunks(chunks, max_len=2000)

    # Prepare collection
    coll = None
    try:
        coll = Collection(COLLECTION_NAME)
    except Exception:
        coll = None

    if coll is None:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1536),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="hash", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=100),
        ]
        schema = CollectionSchema(fields, description="Books RAG collection")
        coll = Collection(name=COLLECTION_NAME, schema=schema)
        print(f"✅ Created collection: {COLLECTION_NAME}")

        coll.create_index(
            field_name="vector",
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 128}
            }
        )
        print("✅ Index created")
        coll.load()
        print("✅ Collection loaded for searching")

    # Deduplicate and insert only new chunks
    new_rows = []
    for chunk in chunks:
        h = chunk_hash(chunk)
        existing = coll.query(expr=f'hash == "{h}"', output_fields=["hash"])
        if existing:
            continue
        embedding = embed_text(chunk)
        new_rows.append({"vector": embedding, "text": chunk, "hash": h, "source": source})

    if new_rows:
        coll.insert(new_rows)
        print(f"✅ Inserted {len(new_rows)} new chunks")
    else:
        print("ℹ️ No new chunks to insert (all duplicates skipped)")
# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    index_pdf("autobiography-of-a-yogi.pdf","Autobiography of a yogi by Paramhamsa Yogananda")
    index_pdf("Reiki Raja Yoga.pdf", "Reiki Raja Yoga by Shaiesh Kumar" )
    index_pdf("C:\learning\dhc documents\divineheartcenter youtube.pdf", "divineheartcenter youtube by Shaiesh Kumar" )
   
