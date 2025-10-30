from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging
import os
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Import fallback ML function
try:
    from app.utils.ml_utils import predict_major
    logger.info("✅ ML fallback imported")
except:
    predict_major = None

# Import RAG engine
try:
    from app.rag_engine import CareerCompassWeaviate
    career_system = CareerCompassWeaviate()
    logger.info("✅ RAG engine imported")
except Exception as e:
    logger.error(f"❌ RAG import failed: {e}")
    career_system = None

# Static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Career Compass...")
    
    if career_system:
        try:
            dataset_path = "app/final_merged_career_guidance.csv"
            if os.path.exists(dataset_path):
                career_system.initialize_system(dataset_path)
                logger.info("✅ RAG system initialized")
            else:
                logger.error("❌ Dataset not found")
        except Exception as e:
            logger.error(f"❌ Startup error: {e}")

@app.get("/")
async def home(request: Request):
    work_styles = ["Team-Oriented","Remote", "On-site","Office/Data", "Hands-on/Field","Lab/Research","Creative/Design", "People-centric/Teaching", "Business", "freelance"]
    return templates.TemplateResponse("index.html", {"request": request, "work_styles": work_styles})

@app.post("/ask")
async def ask_question(data: dict):
    try:
        if career_system and hasattr(career_system, 'is_initialized') and career_system.is_initialized:
            response = career_system.ask_question(data.get("question", ""))
            return {"answer": response["answer"]}
        else:
            return {"answer": "🤖 Career AI is currently upgrading! We're enhancing our career guidance system and will be back with improved recommendations shortly. Thank you for your patience!"}
    except:
        return {"answer": "Career guidance service is temporarily unavailable. Please try again later."}

@app.post("/predict")
async def predict(
    R: str = Form(None), I: str = Form(None), A: str = Form(None),
    S: str = Form(None), E: str = Form(None), C: str = Form(None),
    skills: str = Form(""), courses: str = Form(""),
    work_style: str = Form(""), passion: str = Form("")
):
    if predict_major:
        try:
            riasec = {k: bool(v) for k, v in zip("RIASEC", [R,I,A,S,E,C])}
            user_data = {"riasec": riasec, "skills_text": skills, "courses_text": courses, "work_style": work_style, "passion_text": passion}
            result = predict_major(user_data)
            return JSONResponse(result)
        except Exception as e:
            return JSONResponse({"success": False, "error": str(e)})
    else:
        return JSONResponse({
            "success": True,
            "major": "Computer Science", 
            "faculty": "Faculty of Engineering",
            "degree": "Bachelor of Science",
            "campus": "Main Campus",
            "note": "ML system upgrading - sample recommendation"
        })

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "Career Compass",
        "ml_ready": predict_major is not None,
        "rag_ready": career_system is not None,
        "port": 8080
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
