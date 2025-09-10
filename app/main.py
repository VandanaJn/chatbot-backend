# backend/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from openai import OpenAI
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from app.milvus_client import build_context  # assumes reranking + NumPy cosine is implemented
load_dotenv()

# -------------------------------
# Configuration
# -------------------------------
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

MAX_CONVERSATION_LENGTH = 12  # keep last N messages per user

# Initialize FastAPI + OpenAI client
app = FastAPI()
client = OpenAI(api_key=api_key)

# In-memory conversation store
conversations = {}

# Allow CORS from dev frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------
# Pydantic request model
# -------------------------------
class ChatRequest(BaseModel):
    user_id: str
    text: str


# -------------------------------
# System prompt
# -------------------------------
SYSTEM_PROMPT = (
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


# -------------------------------
# Chat endpoint
# -------------------------------
@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    user_id = request.user_id
    query = request.text

    # Initialize conversation if new user
    if user_id not in conversations:
        conversations[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Fetch context from Milvus with safe error handling
    try:
        context = build_context(query)  # returns string
    except Exception as e:
        return {
            "reply": f"⚠️ Error fetching context: {str(e)}",
            "conversation": conversations.get(user_id, [])
        }

    # If no relevant context, return apology
    if not context.strip():
        return {
            "reply": "I’m sorry, I do not contain sufficient information to answer this question.",
            "conversation": conversations.get(user_id, [])
        }

    # Append user message with context
    user_message = {
        "role": "user",
        "content": (
            f"CONTEXT (from the book):\n{context}\n\n"
            f"QUESTION: {query}\n\n"
            "⚠️ IMPORTANT: Use ONLY the context above. Do not add anything beyond it."
        )
    }
    conversations[user_id].append(user_message)

    # Keep only last N messages to avoid token overflow
    conversations[user_id] = conversations[user_id][-MAX_CONVERSATION_LENGTH:]

    # Generate GPT response
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversations[user_id],
            temperature=0,   # deterministic
            top_p=1,
            max_tokens=500
        )
        reply = response.choices[0].message.content.strip()
    except Exception as e:
        reply = f"⚠️ Error generating response: {str(e)}"

    # Save assistant response
    conversations[user_id].append({"role": "assistant", "content": reply})

    return {
        "reply": reply,
        "conversation": conversations[user_id],
    }


# -------------------------------
# Health check
# -------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -------------------------------
# Run server
# -------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
