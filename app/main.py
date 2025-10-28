from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging
import os
import uvicorn
import sys

# Add current directory to Python path
sys.path.insert(0, '/app')

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize services as None - will be imported later
career_system = None
predict_major = None

def initialize_services():
    global career_system, predict_major
    
    # Import ml_utils with error handling
    try:
        from app.utils.ml_utils import predict_major as pm
        predict_major = pm
        logger.info("✅ ML utils imported successfully")
    except Exception as e:
        logger.error(f"❌ Failed to import ML utils: {e}")
        predict_major = None

    # Import and initialize RAG system
    try:
        from app.rag_engine import CareerCompassWeaviate
        career_system = CareerCompassWeaviate()
        logger.info("✅ RAG engine imported successfully")
        
        # Initialize RAG system with multiple path fallbacks
        dataset_paths = [
            "app/final_merged_career_guidance.csv",
            "final_merged_career_guidance.csv",
            "/app/app/final_merged_career_guidance.csv"
        ]
        
        for path in dataset_paths:
            if os.path.exists(path):
                logger.info(f"📁 Found dataset at: {path}")
                success = career_system.initialize_system(path)
                if success:
                    logger.info("✅ RAG system initialized successfully")
                    break
                else:
                    logger.error(f"❌ Failed to initialize RAG system with {path}")
            else:
                logger.warning(f"📁 Dataset not found at: {path}")
        else:
            logger.error("❌ No dataset file found in any location!")
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {e}")
        career_system = None

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Career Compass services...")
    initialize_services()

# Mount static files and templates
try:
    app.mount("/static", StaticFiles(directory="app/static"), name="static")
    logger.info("✅ Static files mounted")
except Exception as e:
    logger.warning(f"⚠️ Could not mount static files: {e}")

try:
    templates = Jinja2Templates(directory="app/templates")
    logger.info("✅ Templates configured")
except Exception as e:
    logger.error(f"❌ Templates failed: {e}")
    templates = None

@app.get("/health")
def health():
    return {
        "status": "ok",
        "ml_system_ready": predict_major is not None,
        "rag_system_ready": career_system is not None
    }

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    work_styles = ["Team-Oriented","Remote","On-site","Office/Data","Hands-on/Field",
                   "Lab/Research","Creative/Design","People-centric/Teaching",
                   "Business","freelance"]
    
    if templates is None:
        return HTMLResponse("<h1>Career Compass</h1><p>Service starting up...</p>")
    
    return templates.TemplateResponse("index.html", {"request": request, "work_styles": work_styles})

@app.post("/ask")
async def ask_question(data: dict):
    try:
        if career_system is None:
            return {"answer": "Career guidance system is currently unavailable. Please try again later."}
        
        q = data.get("question")
        logger.info(f"Received question: {q}")
        response = career_system.ask_question(q)
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
    if predict_major is None:
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
        logger.error(f"Error in predict endpoint: {e}")
        return JSONResponse({"success": False, "error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080)) 
    uvicorn.run("main:app", host="0.0.0.0", port=port)
