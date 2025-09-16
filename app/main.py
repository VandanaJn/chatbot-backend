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

# SYSTEM_PROMPT = (
#     "You are a witty and compassionate spiritual guru, guiding seekers with ancient wisdom and modern insight. "
#     "Answer questions using retrieved passages as guidance. "
#     "If the passages do not explicitly answer the question, you may reason thoughtfully to provide insights, "
#     "but always anchor your reasoning in the passages. "
#     "Occasionally reference the book name or source once when helpful. "
#     "For simple, everyday questions you may answer naturally in a yogi-style. "
#     "Never say 'document', 'PDF', 'context', 'text', or 'information'."
# )

SYSTEM_PROMPT = (
    "You are a witty and compassionate spiritual guru, guiding seekers with ancient wisdom and modern insight. "
    "Keep answers short and clear, just a few sentences. "
    "Use retrieved passages as guidance. "
    "If the passages do not explicitly answer the question, you may reason briefly, "
    "but always anchor your reasoning in the passages. "
    "Mention the book or source occasionally, but not every time. "
    # "Address the user warmly as a seeker only sometimes, not in every sentence. "
    "For simple, everyday questions you may answer naturally in a yogi-style. "
    "Never say 'document', 'PDF', 'context', 'text', or 'information'."
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

    # Collect recent exchanges for semantic search
    recent_msgs = [
    m for m in conversations[user_id][-RECENT_EXCHANGES_FOR_SEARCH*2:]
    if m["role"] in ("user", "assistant")
    ]

    # --- NEW: Rewrite user query with context ---
    def build_search_query(history, query):
        history_text = " ".join([m["content"] for m in history])
        prompt = (
            f"Conversation so far:\n{history_text}\n\n"
            f"User's new question: {query}\n\n"
            "Rephrase this into a short, standalone query for retrieving relevant passages "
            "from scriptures and spiritual texts."
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You rewrite user queries for retrieval."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=50
        )
        return resp.choices[0].message.content.strip()

    search_query = build_search_query(recent_msgs, query)
    print(search_query)

    # Fetch context from Milvus
    try:
        context = build_context(search_query)  # late-chunk + reranked
    except Exception as e:
        return {
            "reply": f"⚠️ Error fetching context: {str(e)}",
            "conversation": conversations.get(user_id, [])
        }

    # If no context, fallback response
    if not context.strip():
        reply = (
            "I’m sorry, I don’t see a relevant passage, "
            "but in the spirit of the teachings, meditation and self-realization "
            "can guide modern life and personal growth."
        )
        conversations[user_id].append({"role": "assistant", "content": reply})
        return {"reply": reply, "conversation": conversations[user_id]}

    # Compact, reasoning-friendly user message
    user_message = {
        "role": "user",
        "content": (
            f"Passages:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer briefly using the passages as guidance. "
            "If the passages do not explicitly answer, reason thoughtfully based on their principles."
        )
    }
    conversations[user_id].append(user_message)

    # Keep last N messages to limit token usage
    conversations[user_id] = conversations[user_id][-MAX_CONVERSATION_LENGTH:]
    messages_to_send = [conversations[user_id][0]] + conversations[user_id][-MAX_CONVERSATION_LENGTH:]

    # Generate GPT response
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages_to_send,
            temperature=0.2,
            top_p=0.95,
            max_tokens=500
        )
        reply = response.choices[0].message.content.strip()
    except Exception as e:
        reply = f"⚠️ Error generating response: {str(e)}"

    conversations[user_id].append({"role": "assistant", "content": reply})

    return {"reply": reply, "conversation": conversations[user_id]}




@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
