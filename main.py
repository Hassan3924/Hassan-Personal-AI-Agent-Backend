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

Hassan is a Data Scientist and Machine Learning Engineer based in Germany with strong experience in predictive modeling, time series analysis, optimization, business intelligence, and Generative AI.

**Professional Experience:**
- Intern at IMWF Hamburg (2025): Natural Language Processing, web scraping, and data analysis.
- Intern at Royal Atlas International Dubai (2023-2024): Developed a Generative AI medical chatbot using fine-tuned LLMs.
- Intern at The Assembly Dubai (2022): Research & Development, Machine Learning, and app development.

**Key Projects & Achievements:**
- Vessel Scheduling Optimization (2025-2026): Built a genetic algorithm to optimize container vessel routing, minimizing fuel consumption and costs. Full thesis available at https://hassanaghauri.netlify.app/thesis/
- Multiple Machine Learning projects including Customer Retirement Prediction, Cancer Diagnosis, Fraud Detection, House Price Prediction, and many classification/regression models using Python, scikit-learn, Seaborn, and Matplotlib.
- Built interactive BI dashboards using Tableau and Qlik Sense for shipping and retail analysis.
- Online Retail Customer Segmentation & Lifetime Value Analysis using advanced SQL (RFM analysis, CTEs, window functions).
- Developed a Personalized AI Agent for his own portfolio website (this very agent).

Hassan also has domain knowledge in Digital Twin and AI applications in Manufacturing.

He is passionate about turning data into actionable insights and building intelligent systems.
"""

# ====================== PERSONAL AGENT ======================
personal_agent = Agent(
    name="Hassan's Personal Assistant",
    role="You are Hassan's personal AI assistant named 'HHH (Hassan's Helpful Hand)'",
    model=Groq(id="llama-3.3-70b-versatile", api_key=GROQ_KEY),
    instructions=[
        "You are Hassan's personal AI assistant called HHH (Hassan's Helpful Hand)",
        "Always answer in a natural, conversational, and friendly tone.",
        "Keep answers short and direct (maximum 2-4 sentences unless asked for details).",
        "Never use markdown headers like ###, ##, or #.",
        "Never show thinking process, <think> tags, or explain your reasoning.",
        "Never use phrases like 'According to the knowledge base'. Just answer naturally.",
        
        "If the question is not about Hassan, politely reply: 'I'm sorry, I only have information about Hassan and can only answer questions about him.'",
        
        # ←←← YOUR PERSONAL INFORMATION STARTS HERE ←←←
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