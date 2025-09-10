# backend/main.py (multi-turn search with assistant messages)
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from openai import OpenAI
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from app.milvus_client import build_context  # late-chunk + rerank
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

MAX_CONVERSATION_LENGTH = 12      # Keep last N messages
RECENT_EXCHANGES_FOR_SEARCH = 2   # Last N user+assistant exchanges for semantic search

app = FastAPI()
client = OpenAI(api_key=api_key)
conversations = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


class ChatRequest(BaseModel):
    user_id: str
    text: str


@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    user_id = request.user_id
    query = request.text

    # Initialize conversation for new users
    if user_id not in conversations:
        conversations[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Collect last N exchanges (user + assistant) for semantic search
    recent_msgs = conversations[user_id][-RECENT_EXCHANGES_FOR_SEARCH*2:]  # last N exchanges
    # Extract last user message
    last_user_msg = ([m["content"] for m in conversations[user_id] if m["role"] == "user"] or [query])[-1]
    # Combine messages with last user message repeated
    combined_query = " ".join([m["content"] for m in recent_msgs] + [last_user_msg] + [query])

    # Fetch context from Milvus using combined query
    try:
        context = build_context(combined_query)  # returns late-chunked + reranked context string
    except Exception as e:
        return {
            "reply": f"⚠️ Error fetching context: {str(e)}",
            "conversation": conversations.get(user_id, [])
        }

    if not context.strip():
        return {
            # "reply": "I’m sorry, I do not contain sufficient information to answer this question.",
            "reply": "I’m sorry, no matching text.",
            "conversation": conversations.get(user_id, [])
        }

    # Append user message with context
    user_message = {
    "role": "user",
    "content": (
        f"Answer the question using only the text below:\n\n"
        f"{context}\n\n"
        f"Question: {query}\n\n"
        "Do not add anything outside the text. "
        "optionally mention the book name or source once if needed. Avoid saying 'document', 'context', 'information', 'text'."
    )
}
    conversations[user_id].append(user_message)

    # Keep last N messages to avoid token overflow
    conversations[user_id] = conversations[user_id][-MAX_CONVERSATION_LENGTH:]

    # Generate GPT response
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversations[user_id],
            temperature=.09,
            top_p=.95,
            max_tokens=500
        )
        reply = response.choices[0].message.content.strip()
    except Exception as e:
        reply = f"⚠️ Error generating response: {str(e)}"

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
