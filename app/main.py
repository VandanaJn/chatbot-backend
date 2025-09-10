# backend/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from openai import OpenAI
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from app.milvus_client import query_milvus 
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

# Initialize app + OpenAI
app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# In-memory conversation store (simple dictionary)
conversations = {}

   
    
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # frontend dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    user_id: str
    text: str

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    user_id = request.user_id
    query = request.text

    # Get context chunks from Milvus
    context_chunks = query_milvus(query, top_k=3)
    context = "\n".join(context_chunks)

    # If no context found, return apology immediately
    if not context_chunks:
        return {
            "reply": "I’m sorry, I do not contain sufficient information to answer this question.",
            "conversation": conversations.get(user_id, [])
        }

    # If this is a new user, initialize their conversation
    if user_id not in conversations:
        conversations[user_id] = [
            {
                "role": "system",
                "content": (
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
            }
        ]

    # Append query WITH retrieved context
    conversations[user_id].append({
        "role": "user",
        "content": (
            f"CONTEXT (from the book):\n{context}\n\n"
            f"QUESTION: {query}\n\n"
            "⚠️ IMPORTANT: Use ONLY the context above. "
            "Do not add anything beyond it."
        )
    })

    # Generate response from OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversations[user_id],
        temperature=0,   # 🔒 deterministic
        top_p=1,
        max_tokens=500   # optional safety limit
    )

    reply = response.choices[0].message.content.strip()

    # Save assistant response into conversation
    conversations[user_id].append({"role": "assistant", "content": reply})

    return {
        "reply": reply,
        "conversation": conversations[user_id],
    }


@app.get("/health")
def health():
    return {"status": "ok"}
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
