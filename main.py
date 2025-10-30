from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging
import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

career_system = None
predict_major = None

# Import ML utils
try:
    from app.utils.ml_utils import predict_major, load_models
    logger.info("✅ ML utils imported")
except Exception as e:
    logger.error(f"❌ Failed to import ML utils: {e}")
    predict_major = None

# Import RAG engine
try:
    from app.rag_engine import CareerCompassWeaviate
    logger.info("✅ RAG engine imported")
except Exception as e:
    logger.error(f"❌ Failed to import RAG engine: {e}")
    CareerCompassWeaviate = None

# Static files
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
    
    # Debug: Show current directory structure
    logger.info(f"📂 Current directory: {os.getcwd()}")
    logger.info(f"📂 Root contents: {os.listdir('.')}")
    
    if os.path.exists('models'):
        logger.info(f"📁 Models directory contents: {os.listdir('models')}")
    else:
        logger.error("❌ Models directory not found!")
    
    # Load ML models first
    if predict_major:
        try:
            from app.utils.ml_utils import load_models
            ml_loaded = load_models()
            if ml_loaded:
                logger.info("✅ ML models loaded successfully")
            else:
                logger.error("❌ Failed to load ML models")
        except Exception as e:
            logger.error(f"❌ Error loading ML models: {e}")
            import traceback
            logger.error(f"🔍 ML loading traceback: {traceback.format_exc()}")
    
    # Initialize RAG system
    if CareerCompassWeaviate:
        try:
            career_system = CareerCompassWeaviate()
            logger.info("✅ Career system instance created")
            
            main_path = "app/final_merged_career_guidance.csv"
            
            if os.path.exists(main_path):
                logger.info(f"📁 Found dataset at: {main_path}")
                success = career_system.initialize_system(main_path)
                if success:
                    logger.info("✅ Career Compass RAG initialized successfully!")
                else:
                    logger.error("❌ RAG initialization failed")
            else:
                logger.error(f"❌ Dataset not found at: {main_path}")
                    
        except Exception as e:
            logger.error(f"❌ Error initializing Career Compass: {e}")
            import traceback
            logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
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
        if not career_system or not hasattr(career_system, 'is_initialized') or not career_system.is_initialized:
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
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"Error in predict: {e}")
        return JSONResponse({"success": False, "error": str(e)})

@app.get("/health")
async def health():
    dataset_exists = os.path.exists("app/final_merged_career_guidance.csv")
    rag_ready = career_system is not None and hasattr(career_system, 'is_initialized') and career_system.is_initialized
    
    # Check if ML models are loaded
    ml_ready = False
    try:
        from app.utils.ml_utils import model
        ml_ready = model is not None
    except:
        pass

    return {
        "status": "healthy",
        "service": "Career Compass",
        "ml_ready": ml_ready,
        "rag_ready": rag_ready,
        "dataset_available": dataset_exists,
        "port": 8080
    }

@app.get("/debug/models")
async def debug_models():
    """Debug endpoint to check model files"""
    import glob
    models_info = {}
    
    # Check models directory
    if os.path.exists("models"):
        model_files = glob.glob("models/*.pkl")
        models_info["model_files"] = model_files
        models_info["models_dir_exists"] = True
        models_info["models_dir_path"] = os.path.abspath("models")
    else:
        models_info["models_dir_exists"] = False
        models_info["model_files"] = []
        models_info["models_dir_path"] = "Not found"
    
    # Check if ML function is available
    models_info["predict_major_available"] = predict_major is not None
    
    # Check current directory
    models_info["current_directory"] = os.getcwd()
    models_info["root_contents"] = os.listdir('.')
    
    return JSONResponse(models_info)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
