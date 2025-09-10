from openai import OpenAI
from pymilvus import (
    connections,
    Collection
)
import numpy as np

# -------------------------------
# Configuration
# -------------------------------
COLLECTION_NAME = "books_rag"

# Connect to Milvus
connections.connect(alias="default", host="localhost", port="19530")

# Initialize OpenAI
openai_client = OpenAI()


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

def late_chunk(text: str, chunk_size: int = 400, overlap: int = 50):
    """
    Re-chunk text into smaller overlapping slices for final answer generation.
    """
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# -------------------------------
# Query Milvus
# -------------------------------
def query_milvus(query: str, top_k: int = 7, min_score: float = 0.5):
    coll = Collection(COLLECTION_NAME)
    query_emb = embed_text(query)

    results = coll.search(
        data=[query_emb],
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["text", "source"]
    )

    hits = results[0]
    res = []
    for hit in hits:
        if hit.score >= min_score:
            res.append({
                "text": hit.entity.get("text", ""),
                "source": hit.entity.get("source", "unknown source")
            })

    return res

def rerank_late_chunks(query: str, late_chunks: list, top_n: int = 8):
    """
    Re-rank late chunks by embedding similarity to the query.
    Returns only the top_n most relevant.
    """
    query_emb = embed_text(query)

    scored_chunks = []
    for item in late_chunks:
        emb = embed_text(item["text"])  # embed each late chunk
        # cosine similarity manually
        score = cosine_similarity(query_emb, emb)
        scored_chunks.append((score, item))

    # Sort by similarity
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    return [item for _, item in scored_chunks[:top_n]]



# -------------------------------
# Generate answer using GPT
# -------------------------------
def generate_answer(query: str, top_n_late: int = 8):
    context = build_context(query, top_n_late)

    system_prompt = (
        "You are a spiritual guru guiding a beginner on their spiritual journey.\n"
        "⚠️ RULES:\n"
        "1. Answer ONLY using the provided context.\n"
        "2. Do NOT use outside knowledge or add anything extra.\n"
        "3. If the context does not contain the answer, respond exactly with:\n"
        "   'I’m sorry, I do not contain sufficient information to answer this question.'\n"
        "4. Always reference the book name if available.\n"
        "5. Mention the book name only once per response.\n"
        "6. Never say 'document' or 'PDF'.\n"
    )

    user_prompt = f"Answer the question using ONLY the context below:\n\n{context}\n\nQuestion: {query}\nAnswer:"

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content

def build_context(query, top_n_late=8):
    retrieved = query_milvus(query)

    # Step 1: re-slice into late chunks
    late_chunks = []
    for item in retrieved:
        for lc in late_chunk(item["text"], chunk_size=400, overlap=50):
            late_chunks.append({"text": lc, "source": item["source"]})

    # Step 2: rerank and keep only top_n_late chunks
    best_chunks = rerank_late_chunks(query, late_chunks, top_n=top_n_late)

    # Step 3: build context
    context = "\n".join(
        f"{chunk['text']} (from source: {chunk['source']})"
        for chunk in best_chunks
    )
    
    return context

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    
    answer = generate_answer("Is reiki unconditional love?")
    print("🤖 Answer:", answer)
