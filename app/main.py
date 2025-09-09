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

    # If this is a new user, initialize their conversation
    if user_id not in conversations:
        conversations[user_id] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Answer ONLY using the provided context from the PDF/book. "
                    "If the answer is not in the context, say 'I don’t know from the document.'"
                ),
            }
        ]

    # Append the new user query
    conversations[user_id].append({"role": "user", "content": query})

    # Add retrieved context as a hidden system guidance message
    conversations[user_id].append({"role": "system", "content": f"Context from PDF:\n{context}"})

    # Generate response using OpenAI with conversation + PDF context
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversations[user_id],
    )

    reply = response.choices[0].message.content

    # Save assistant response into conversation history
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
