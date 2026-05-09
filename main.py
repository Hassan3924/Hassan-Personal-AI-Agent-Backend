from agno.agent import Agent
from agno.models.groq import Groq
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

model_name = "llama-3.1-8b-instant"

load_dotenv()

GROQ_KEY = os.getenv('GROQ_API_KEY')

PERSONAL_INFO = """
You are the personal AI assistant of Hassan Abdullah Ghauri.

Hassan is a Data Scientist and Machine Learning Engineer based in Germany. He has worked on numerous data science, machine learning, and AI projects, with a strong focus on predictive modeling, optimization, time series analysis, and building customer-centric AI solutions.

**Standout Projects:**

- **Customer No-Show Prediction**: Developed a highly accurate predictive model for a liner shipping company, achieving over 90% accuracy through rigorous model evaluation and feature engineering. 
  Link: https://hassanaghauri.netlify.app/noshowprediction/

- **Vessel Scheduling Optimization**: Built a genetic algorithm to optimize container vessel routing and scheduling. The solution minimized fuel consumption and voyage costs while maintaining perfect schedule reliability. 
  Link: https://hassanaghauri.netlify.app/thesis/

- **Germany Tourism Projection**: Conducted time series forecasting to project tourism trends in Germany.

- **Weather Data Collection & Pipeline**: Designed and implemented an automated ETL pipeline using Airflow, Docker, and AWS to collect and process large-scale weather data.

- **Online Retail Customer Segmentation & Lifetime Value Analysis**: Performed advanced SQL analysis (RFM, CTEs, window functions) on a 541,000+ transaction dataset to deliver actionable customer insights and segmentation strategies.

- **Personalized AI Agent**: Currently developing and maintaining a personalized AI agent (this very assistant) for his portfolio website to answer questions about his background, skills, and projects. He is actively working on building more customer-centric AI agents in parallel.

Hassan also has practical experience in Natural Language Processing, Generative AI, Business Intelligence (Tableau & Qlik Sense), and domain knowledge in Digital Twin and AI applications in Manufacturing.

He is passionate about turning complex data into actionable insights and creating intelligent, user-focused solutions.
"""

# ====================== PERSONAL AGENT ======================
personal_agent = Agent(
    name="Hassan's Personal Assistant",
    role="You are Hassan's personal AI assistant.",
    model=Groq(id="llama-3.3-70b-versatile", api_key=GROQ_KEY),
    instructions=[
        "You are Hassan's personal AI assistant.",
        
        "Answer in a **natural, conversational tone** — like a friendly human speaking.",
        "Do NOT use bold text (** **) excessively.",
        "Do NOT use too many markdown formatting symbols.",
        "Use bold **only** when it is really necessary (maximum once or twice per response).",
        "Prefer plain, simple text most of the time.",
        "Keep answers short, direct, and easy to read.",
        "Never use headings like ### or ##.",
        "Never show thinking process or <think> tags.",
        
        "If the question is not about Hassan, politely reply: 'I'm sorry, I only have information about Hassan and can only answer questions about him.'",
        
        PERSONAL_INFO,
    ],
    markdown=True
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