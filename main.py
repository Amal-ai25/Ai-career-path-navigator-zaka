from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging
import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Import from app folder
try:
    from app.utils.ml_utils import predict_major
    logger.info("✅ ML utils imported")
except Exception as e:
    logger.error(f"❌ ML utils import failed: {e}")
    predict_major = None

try:
    from app.rag_engine import CareerCompassWeaviate
    logger.info("✅ RAG engine imported")
except Exception as e:
    logger.error(f"❌ RAG engine import failed: {e}")
    CareerCompassWeaviate = None

# Mount static files from app folder
try:
    app.mount("/static", StaticFiles(directory="app/static"), name="static")
    templates = Jinja2Templates(directory="app/templates")
    logger.info("✅ Static files configured")
except Exception as e:
    logger.error(f"❌ Static files error: {e}")
    templates = None

career_system = None

@app.on_event("startup")
async def startup_event():
    global career_system
    logger.info("🚀 Starting services...")
    
    if CareerCompassWeaviate:
        try:
            career_system = CareerCompassWeaviate()
            csv_path = "final_merged_career_guidance.csv"
            if os.path.exists(csv_path):
                success = career_system.initialize_system(csv_path)
                if success:
                    logger.info("✅ System initialized")
                else:
                    logger.error("❌ System init failed")
            else:
                logger.error(f"❌ Dataset not found: {csv_path}")
        except Exception as e:
            logger.error(f"❌ Startup error: {e}")

@app.get("/")
async def home(request: Request):
    if templates:
        work_styles = ["Team-Oriented","Remote","On-site","Office/Data","Hands-on/Field","Lab/Research","Creative/Design","People-centric/Teaching","Business","freelance"]
        return templates.TemplateResponse("index.html", {"request": request, "work_styles": work_styles})
    else:
        return HTMLResponse("<h1>Career Compass</h1><p>Service starting...</p>")

# ... rest of your routes remain the same

@app.post("/ask")
async def ask_question(data: dict):
    try:
        if not career_system:
            return {"answer": "Career guidance system is currently unavailable. Please try again later."}
        
        q = data.get("question", "").strip()
        if not q:
            return {"answer": "Please provide a question."}
            
        response = career_system.ask_question(q)
        return {"answer": response["answer"]}
    except Exception as e:
        logger.error(f"Error in ask_question: {e}")
        return {"answer": "Sorry, I'm having trouble processing your question right now."}

@app.post("/predict")
async def predict(
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
    if not predict_major:
        return JSONResponse({"success": False, "error": "Prediction system is currently unavailable."})
    
    try:
        riasec = {k: bool(v) for k, v in zip("RIASEC", [R,I,A,S,E,C])}
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
        logger.error(f"Error in predict: {e}")
        return JSONResponse({"success": False, "error": str(e)})

# Health check endpoint for Render
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "Career Compass",
        "ml_ready": predict_major is not None,
        "rag_ready": career_system is not None
    }

@app.get("/test")
async def test():
    return {"message": "Server is running!", "timestamp": "now"}

# Render runs this automatically - no need for __main__ block
