from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging
import os
import sys

# Add current directory to Python path
sys.path.insert(0, '/app')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize with error handling
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

# Lazy imports with error handling
predict_major = None
CareerCompassWeaviate = None
career_system = None

def initialize_services():
    global predict_major, CareerCompassWeaviate, career_system
    
    # Import ml_utils with error handling
    try:
        from app.utils.ml_utils import predict_major as pm
        predict_major = pm
        logger.info("✅ ML utils imported successfully")
    except Exception as e:
        logger.error(f"❌ Failed to import ML utils: {e}")
        predict_major = None

    # Import rag_engine with error handling  
    try:
        from app.rag_engine import CareerCompassWeaviate as CCW
        CareerCompassWeaviate = CCW
        logger.info("✅ RAG engine imported successfully")
    except Exception as e:
        logger.error(f"❌ Failed to import RAG engine: {e}")
        CareerCompassWeaviate = None

    # Initialize RAG system
    if CareerCompassWeaviate:
        try:
            career_system = CareerCompassWeaviate()
            dataset_path = "app/final_merged_career_guidance.csv"
            if os.path.exists(dataset_path):
                success = career_system.initialize_system(dataset_path)
                if success:
                    logger.info("✅ RAG system initialized successfully")
                else:
                    logger.error("❌ RAG system initialization failed")
                    career_system = None
            else:
                logger.error(f"❌ Dataset not found at: {dataset_path}")
                career_system = None
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG system: {e}")
            career_system = None

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Career Compass services...")
    initialize_services()

@app.get("/health")
async def health():
    status = {
        "status": "healthy",
        "ml_system_ready": predict_major is not None,
        "rag_system_ready": career_system is not None
    }
    return status

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    work_styles = [
        "Team-Oriented", "Remote", "On-site", "Office/Data", "Hands-on/Field",
        "Lab/Research", "Creative/Design", "People-centric/Teaching",
        "Business", "freelance"
    ]
    
    if templates is None:
        return HTMLResponse("<h1>Career Compass</h1><p>Template system not available</p>")
    
    return templates.TemplateResponse("index.html", {"request": request, "work_styles": work_styles})

@app.post("/ask")
async def ask_question(data: dict):
    if career_system is None:
        return {"answer": "Career guidance system is currently unavailable. Please try again later."}
    
    try:
        q = data.get("question", "").strip()
        if not q:
            return {"answer": "Please provide a question."}
            
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
        riasec = {
            "R": bool(R), "I": bool(I), "A": bool(A),
            "S": bool(S), "E": bool(E), "C": bool(C)
        }
        
        user_data = {
            "riasec": riasec,
            "skills_text": skills,
            "courses_text": courses,
            "work_style": work_style,
            "passion_text": passion
        }

        result = predict_major(user_data)
        
        if "error" in result:
            return JSONResponse({"success": False, "error": result["error"]})
        else:
            result["success"] = True
            return JSONResponse(result)
            
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        return JSONResponse({"success": False, "error": str(e)})

# Simple test endpoint
@app.get("/test")
async def test():
    return {
        "message": "Career Compass API is running",
        "ml_available": predict_major is not None,
        "rag_available": career_system is not None
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
