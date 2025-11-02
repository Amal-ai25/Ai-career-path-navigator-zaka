from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging
import os
import uvicorn
import sys

# Add parent directory to Python path to fix imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize services
career_system = None
predict_major = None

def initialize_services():
    global career_system, predict_major
    
    # Import ML system
    try:
        # Use relative import since we're in app/ directory
        from .utils.ml_utils import predict_major as pm
        predict_major = pm
        logger.info("✅ ML system imported successfully")
    except Exception as e:
        logger.error(f"❌ Failed to import ML utils: {e}")
        predict_major = None

    # Import RAG system
    try:
        # Use relative import
        from .rag_engine import CareerCompassWeaviate
        career_system = CareerCompassWeaviate()
        logger.info("✅ RAG system created successfully")
        
        # Try multiple dataset paths
        dataset_paths = [
            os.path.join(current_dir, "final_merged_career_guidance.csv"),
            os.path.join(parent_dir, "final_merged_career_guidance.csv"),
            "final_merged_career_guidance.csv"
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
            logger.error("❌ No dataset file found!")
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {e}")
        career_system = None

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Career Compass services...")
    initialize_services()

# Mount static files and templates
try:
    static_dir = os.path.join(current_dir, "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        logger.info("✅ Static files mounted")
    else:
        logger.warning(f"⚠️ Static directory not found: {static_dir}")
except Exception as e:
    logger.warning(f"⚠️ Could not mount static files: {e}")

try:
    templates_dir = os.path.join(current_dir, "templates")
    if os.path.exists(templates_dir):
        templates = Jinja2Templates(directory=templates_dir)
        logger.info("✅ Templates configured")
    else:
        logger.error(f"❌ Templates directory not found: {templates_dir}")
        templates = None
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
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
