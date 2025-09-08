import pdfplumber
from openai import OpenAI
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType
)

# -------------------------------
# Configuration
# -------------------------------
COLLECTION_NAME = "resume_rag"

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

# -------------------------------
# Index PDF into Milvus
# -------------------------------
RECREATE_COLLECTION = False  # Set True to drop & recreate, False to keep existing collection

def index_pdf(pdf_path: str):
    text = read_pdf(pdf_path)
    chunks = chunk_text(text)

    # Generate embeddings
    rows = []
    for chunk in chunks:
        embedding = embed_text(chunk)
        rows.append({"vector": embedding, "text": chunk})

    # Drop collection if RECREATE_COLLECTION is True
    coll = None
    try:
        coll = Collection(COLLECTION_NAME)
        if RECREATE_COLLECTION:
            coll.drop()
            print(f"🗑 Dropped existing collection: {COLLECTION_NAME}")
            coll = None
    except Exception:
        coll = None  # Collection does not exist

    # Create collection if it doesn't exist
    if coll is None:
        from pymilvus import CollectionSchema, FieldSchema, DataType

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=len(rows[0]["vector"])),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
        ]
        schema = CollectionSchema(fields, description="Resume RAG collection")
        coll = Collection(name=COLLECTION_NAME, schema=schema)
        print(f"✅ Created collection: {COLLECTION_NAME}")

        # Create index
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

    # Insert new rows
    coll.insert(rows)
    print(f"✅ Inserted {len(rows)} chunks")


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
        output_fields=["text"]
    )
    hits = results[0]
    return [hit.entity.get("text", "") for hit in hits]

# -------------------------------
# Generate answer using GPT
# -------------------------------
def generate_answer(query: str):
    context_chunks = query_milvus(query)
    context = "\n".join(context_chunks)
    system_prompt = """
        You are a helpful assistant. Answer questions **only using the provided context**.
        Do NOT use any outside knowledge. If the context does not contain the answer, respond with:
        'I’m sorry, the provided documents do not contain sufficient information to answer this question.'
        Always reference the chunk or source if available.
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
    # index_pdf("autobiography-of-a-yogi.pdf")
    answer = generate_answer("tell me about Lahiri mayasaya's first meeting with babaji")
    print("🤖 Answer:", answer)
