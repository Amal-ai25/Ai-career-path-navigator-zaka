from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging
import os
import uvicorn

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from app.utils.ml_utils import predict_major
from app.rag_engine import CareerCompassWeaviate

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# RAG system - lazy loaded
career_system = None

def get_career_system():
    global career_system
    if career_system is None:
        logger.info("Initializing Career Compass RAG system...")
        career_system = CareerCompassWeaviate()
        career_system.initialize_system("app/final_merged_career_guidance.csv")
        logger.info("Career Compass RAG system ready.")
    return career_system

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    work_styles = ["Team-Oriented","Remote","On-site","Office/Data","Hands-on/Field",
                   "Lab/Research","Creative/Design","People-centric/Teaching",
                   "Business","freelance"]
    return templates.TemplateResponse("index.html", {"request": request, "work_styles": work_styles})

@app.post("/ask")
async def ask_question(data: dict):
    try:
        q = data.get("question")
        logger.info(f"Received question: {q}")
        system = get_career_system()
        response = system.ask_question(q)
        return {"answer": response["answer"]}
    except Exception as e:
        logger.error(f"Error in ask_question: {e}")
        return {"answer": "Sorry, I'm having trouble processing your question right now."}

@app.post("/predict")
async def predict(
    request: Request,
    R: str = Form(None),
    I: str = Form(None),
    A: str = Form(None),
    S: str = Form(None),
    E: str = Form(None),
    C: str = Form(None),
    skills: str = Form(""),
    courses: str = Form(""),
    work_style: str = Form(""),
    passion: str = Form("")
):
    try:
        riasec = {k: bool(v) for k, v in zip("RIASCE", [R,I,A,S,E,C])}
        user_data = {
            "riasec": riasec,
            "skills_text": skills,
            "courses_text": courses,
            "work_style": work_style,
            "passion_text": passion
        }
        result = predict_major(user_data)
        result["success"] = "error" not in result
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        return JSONResponse({"success": False, "error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
