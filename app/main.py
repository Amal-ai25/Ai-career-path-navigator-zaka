from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.utils.ml_utils import predict_major
import logging
import os
import uvicorn
from dotenv import load_dotenv
from app.rag_engine import CareerCompassWeaviate

load_dotenv()
# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")
career_system = None

@app.on_event("startup")
def startup_event():
    global career_system
    try:
        career_system = CareerCompassWeaviate()
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_merged_career_guidance.csv")
        success = career_system.initialize_system(csv_path)
        if success:
            logger.info("Career Compass system initialized successfully")
        else:
            logger.error("Career Compass system failed to initialize")
    except Exception as e:
        logger.error(f"Error initializing Career Compass system: {e}")

# Home page - ML Recommendation
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    work_styles = ["Team-Oriented","Remote", "On-site","Office/Data", "Hands-on/Field","Lab/Research","Creative/Design", "People-centric/Teaching", "Business", "freelance"]
    return templates.TemplateResponse("index.html", {"request": request, "work_styles": work_styles})

def get_career_system():
    global career_system
    if career_system is None:
        try:
            from app.rag_engine import CareerCompassWeaviate
            career_system = CareerCompassWeaviate()
            # Don't initialize full system on import to save memory
        except Exception as e:
            logger.error(f"Failed to initialize career system: {e}")
    return career_system

# Chatbot API
@app.post("/ask")
async def ask_question(data: dict):
    try:
        q = data.get("question")
        logger.info(f"Received question: {q}")
        if not career_system:
            return {"answer": "Career system not initialized.", "confidence": "Error"}
        response = career_system.ask_question(q)
        return {"answer": response["answer"]}
    except Exception as e:
        logger.error(f"Error in ask_question: {e}")
        return {"answer": "Sorry, I'm having trouble processing your question right now."}

# ML Prediction Endpoint
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
        logger.info(f"Received prediction request: R={R}, I={I}, A={A}, S={S}, E={E}, C={C}")
        logger.info(f"Skills: {skills}, Courses: {courses}, Work Style: {work_style}, Passion: {passion}")
        
        riasec = {"R": bool(R), "I": bool(I), "A": bool(A), "S": bool(S), "E": bool(E), "C": bool(C)}
        user_data = {
            "riasec": riasec,
            "skills_text": skills,
            "courses_text": courses,
            "work_style": work_style,
            "passion_text": passion
        }

        result = predict_major(user_data)
        logger.info(f"Prediction result: {result}")
        
        if "error" in result:
            return JSONResponse({"success": False, "error": result["error"]})
        else:
            result["success"] = True
            return JSONResponse(result)
            
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        return JSONResponse({"success": False, "error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Changed to 8080
    uvicorn.run("main:app", host="0.0.0.0", port=port)
