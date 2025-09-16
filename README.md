# Spiritual RAG Chatbot Backend

A FastAPI-powered backend for a Retrieval-Augmented Generation (RAG) chatbot designed to provide insightful, scripture-based responses to user queries.

## 🧠 Overview

This backend leverages OpenAI's GPT models, Milvus for vector storage, and FastAPI for real-time interactions.  
It supports multi-turn conversations, contextual search, and maintains a conversation history for personalized responses.

## ⚙️ Features

- **Multi-turn Conversations:** Remembers user interactions to maintain context.
- **Scripture-based Responses:** Fetches relevant passages to answer user queries.
- **Real-time Interaction:** Built with FastAPI for quick responses.
- **Conversation Memory:** Stores user conversations for continuity.

## 🛠️ Requirements

- Python 3.8+  
- FastAPI  
- OpenAI SDK  
- Milvus (vector database) – **must be running locally on `localhost:19530`**  
- Dotenv  
- Uvicorn  

## 🔧 Setup Instructions

Follow these steps to get your chatbot backend running locally.

### 1. Clone the Repository

```bash
git clone https://github.com/VandanaJn/chatbot-backend.git
cd chatbot-backend
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the root directory and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key
```

### 4. Run Milvus Locally (Docker Recommended)

```bash
docker run -d --name milvus \
  -p 19530:19530 -p 19121:19121 \
  milvusdb/milvus:v2.3.0
```

Verify Milvus is running:

```bash
docker ps
```

> ⚠️ The backend assumes Milvus is available at `localhost:19530`.  
> If your host or port differs, update `app/milvus_client.py` accordingly.

### 5. Launch the FastAPI Application

```bash
uvicorn main:app --reload
```

---

## 📚 Document Management

Manage spiritual PDFs in the vector store using these utilities.

### 🧹 Clean Existing Data

```bash
python milvus_cleanup.py
```

Clears all indexed documents from Milvus. Use this before reindexing.

### 📥 Index a New PDF

```python
# index_pdf.py

if __name__ == "__main__":
    index_pdf(
        "Autobiography_of_a_Yogi.pdf",
        "Autobiography of a Yogi by Paramahansa Yogananda"
    )
```

Loads the PDF, chunks it, embeds it, and stores it in Milvus with metadata.

> ✅ Ensure Milvus is running at `localhost:19530` before using either script.
## 🔁 Endpoints

### `POST /chat`

Accepts user queries and returns scripture-based responses.

#### 📥 Request Body

```json
{
  "user_id": "unique_user_identifier",
  "text": "Your question here"
}
```

#### 📤 Response

```json
{
  "reply": "Generated response based on scripture",
  "conversation": [
    { "role": "system", "content": "System message" },
    { "role": "user", "content": "User message" },
    { "role": "assistant", "content": "Assistant message" }
  ]
}
```

---

## 🧠 How It Works

1. **User Input** — The seeker submits a query via `/chat`  
2. **Query Rewriting** — The backend reformulates the query for more accurate retrieval  
3. **Context Retrieval** — Relevant passages are fetched from Milvus  
4. **Response Generation** — GPT generates answers based on retrieved context  
5. **Conversation Memory** — Stores interactions for multi-turn dialogue

---

## 🧪 Development Notes

- Ensure Milvus is running locally  
- Set environment variables via `.env`  
- Uses **late-chunking** and **reranking** to optimize retrieval for multi-turn chat

---

## 📜 License

MIT License. See `LICENSE` for details.