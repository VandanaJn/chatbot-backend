# backend/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from openai import OpenAI
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

# Initialize app + OpenAI
app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# In-memory conversation store (simple dictionary)
conversations = {}

class Message(BaseModel):
    user_id: str  # to separate conversations
    text: str
    
    
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # frontend dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
def chat(msg: Message):
    # user_id = msg.user_id
    user_id="vj"
    

    # Create a new conversation if first time
    if user_id not in conversations:
        conversations[user_id] = [
            {"role": "system", "content": "You are a helpful AI assistant."}
        ]

    # Add user message
    conversations[user_id].append({"role": "user", "content": msg.text})

    # Call OpenAI with full history
    response = client.chat.completions.create(
        model="gpt-4o-mini",  
        messages=conversations[user_id],
    )

    reply = response.choices[0].message.content

    # Add assistant reply to memory
    conversations[user_id].append({"role": "assistant", "content": reply})

    return {"reply": reply}

@app.get("/health")
def health():
    return {"status": "ok"}
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
