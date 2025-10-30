from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging
import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize services
career_system = None
predict_major = None

# Try to import dependencies with error handling
try:
    from app.utils.ml_utils import predict_major
    logger.info("✅ ML utils imported")
except Exception as e:
    logger.error(f"❌ Failed to import ML utils: {e}")
    predict_major = None

try:
    from app.rag_engine import CareerCompassWeaviate
    logger.info("✅ RAG engine imported")
except Exception as e:
    logger.error(f"❌ Failed to import RAG engine: {e}")
    CareerCompassWeaviate = None

# Configure static files and templates
try:
    app.mount("/static", StaticFiles(directory="app/static"), name="static")
    templates = Jinja2Templates(directory="app/templates")
    logger.info("✅ Static files configured")
except Exception as e:
    logger.error(f"❌ Static files error: {e}")
    templates = None

@app.on_event("startup")
async def startup_event():
    global career_system
    logger.info("🚀 Starting Career Compass services...")
    
    if CareerCompassWeaviate:
        try:
            career_system = CareerCompassWeaviate()
            logger.info("✅ Career system instance created")
            
            # Try multiple dataset paths - CSV is in app folder
            dataset_paths = [
                "app/final_merged_career_guidance.csv",  # Main path - CSV in app folder
                "./app/final_merged_career_guidance.csv",
                "final_merged_career_guidance.csv",
                "./final_merged_career_guidance.csv"
            ]
            
            dataset_found = False
            for path in dataset_paths:
                if os.path.exists(path):
                    logger.info(f"📁 Found dataset at: {path}")
                    success = career_system.initialize_system(path)
                    if success:
                        logger.info("✅ Career Compass system initialized successfully")
                        dataset_found = True
                        break
                    else:
                        logger.error(f"❌ Failed to initialize with {path}")
                else:
                    logger.warning(f"📁 Dataset not found at: {path}")
            
            if not dataset_found:
                logger.error("❌ No dataset found in any location. Chat features will be disabled.")
                
        except Exception as e:
            logger.error(f"❌ Error initializing Career Compass system: {e}")
    else:
        logger.error("❌ CareerCompassWeaviate not available")

@app.get("/")
async def home(request: Request):
    if templates:
        work_styles = ["Team-Oriented","Remote", "On-site","Office/Data", "Hands-on/Field","Lab/Research","Creative/Design", "People-centric/Teaching", "Business", "freelance"]
        return templates.TemplateResponse("index.html", {"request": request, "work_styles": work_styles})
    else:
        return HTMLResponse("<h1>Career Compass</h1><p>Service starting up...</p>")

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

@app.get("/health")
async def health():
    # Check if dataset exists in app folder
    dataset_exists = os.path.exists("app/final_merged_career_guidance.csv")
    
    return {
        "status": "healthy",
        "service": "Career Compass",
        "ml_ready": predict_major is not None,
        "rag_ready": career_system is not None,
        "dataset_available": dataset_exists,
        "port": 8080
    }

@app.get("/test")
async def test():
    return {"message": "Server is running!", "timestamp": "now"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
