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

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """Split text into overlapping chunks."""
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

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
    """Generate a SHA256 hash for a text chunk."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# -------------------------------
# Index PDF into Milvus
# -------------------------------
def index_pdf(pdf_path: str, source: str):
    text = read_pdf(pdf_path)
    chunks = chunk_text(text)

    # Prepare collection
    coll = None
    try:
        coll = Collection(COLLECTION_NAME)
    except Exception:
        coll = None

    if coll is None:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=len(embed_text(chunks[0]))),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="hash", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=100),
        ]
        schema = CollectionSchema(fields, description="Resume RAG collection")
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
        if existing:  # hash already exists
            continue
        embedding = embed_text(chunk)
        new_rows.append({"vector": embedding, "text": chunk, "hash": h, "source": source})

    if new_rows:
        coll.insert(new_rows)
        print(f"✅ Inserted {len(new_rows)} new chunks")
    else:
        print("ℹ️ No new chunks to insert (all duplicates skipped)")


# -------------------------------
# Query Milvus
# -------------------------------
def query_milvus(query: str, top_k: int = 3):
    coll = Collection(COLLECTION_NAME)
    query_emb = embed_text(query)
    results = coll.search(
        data=[query_emb],
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["text","source"]
    )
    hits = results[0]
    res= [hit.entity.get("text", "") +f'from source: {hit.entity.get("text", "source")}' for hit in hits]
    print(res)
    return res
# -------------------------------
# Generate answer using GPT
# -------------------------------
def generate_answer(query: str):
    context_chunks = query_milvus(query)
    context = "\n".join(context_chunks)
    system_prompt = """
        You are a helpful assistant. Answer questions **only using the provided context**.
        Do NOT use any outside knowledge. If the context does not contain the answer, respond with:
        'I’m sorry, I do not contain sufficient information to answer this question.'
        cite the source if avilable.
        """

    user_prompt = f"Answer the question using ONLY the context below:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    )
    return response.choices[0].message.content

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    index_pdf("autobiography-of-a-yogi.pdf","Autobiography of a yogi by Paramhamsa Yogananda")
    index_pdf("Reiki Raja Yoga.pdf", "Reiki Raja Yoga by Shaiesh Kumar" )
    answer = generate_answer("what is Reiki raja yoga")
    print("🤖 Answer:", answer)
