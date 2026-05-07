from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.team import Team
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# knowledge base imports
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.chroma import ChromaDb
from agno.knowledge.embedder.google import GeminiEmbedder
from agno.models.google import Gemini
from agno.vectordb.search import SearchType
import os

model_name = "qwen/qwen3-32b"

load_dotenv()

GROQ_KEY = os.getenv('GROQ_API_KEY')
model_name = "qwen/qwen3-32b"

# Knowledge Base
knowledge = Knowledge(
    vector_db = ChromaDb(
        path = "knowledge.chroma",
        collection="hassan_knowledge",
        path = "tmp/chroma_db",
        persistent_client=True,
        search_type=SearchType.hybrid,
        embedder=GeminiEmbedder(id="gemini-embedding-001")
    )
)

knowledge.insert(path = "knowledge/")


#Personal Agent
personal_agent = Agent(
    name = "Hassan's Personal Assistant",
    role = "You are Hassan's personal AI Assistant",
    model = Groq(id = model_name, api_key=GROQ_KEY),
    instructions=["You are Hassan's personal AI assistant.",
        "Answer ONLY using the information from the knowledge base.",
        "If the question is not about Hassan, politely say: 'I'm sorry, I only have information about Hassan and can only answer questions about him.'",
        "Be helpful, friendly, and professional."],
    knowledge=knowledge,
    search_knowledge=True
    markdown = True
)

app = FastAPI(title = "Hassan's Personal AI Assistant Backend")

# Allowing Netilfy website to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://hassanaghauri.netlify.app", 
                   "http://localhost:3000", 
                   "http://localhost:8001",
                   "http://127.0.0.1:8001",
                   "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        result = personal_agent.run(input=request.message)
        # Safely extract the actual text message
        if hasattr(result, "content") and result.content:
            response = result.content
            print(response)
        else:
            response = str(result)

        print("AI Response:", response)

        return {"response": response}

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

# Health check (optional)
@app.get("/")
async def root():
    return {"status": "Hassan's AI Agent is running ✅"}