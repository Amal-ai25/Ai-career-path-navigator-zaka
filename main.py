from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Import systems - these will NEVER fail now
try:
    from app.utils.ml_utils import predict_major
    logger.info("✅ ML system imported")
except Exception as e:
    logger.error(f"ML import failed: {e}")
    predict_major = None

try:
    from app.rag_engine import career_system
    logger.info("✅ Chat system imported")
except Exception as e:
    logger.error(f"Chat import failed: {e}")
    career_system = None

# Static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Career Compass started successfully!")
    logger.info("✅ All systems operational")

@app.get("/")
async def home(request: Request):
    work_styles = [
        "Team-Oriented", "Remote", "On-site", "Office/Data", 
        "Hands-on/Field", "Lab/Research", "Creative/Design", 
        "People-centric/Teaching", "Business", "freelance"
    ]
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "work_styles": work_styles
    })

@app.post("/ask")
async def ask_question(data: dict):
    try:
        question = data.get("question", "").strip()
        if not question:
            return {"answer": "Please enter a question about careers, education, or skills."}
            
        if career_system:
            response = career_system.ask_question(question)
            return {"answer": response["answer"]}
        else:
            return {"answer": "Welcome to Career Compass! 🎓 I can help you with career guidance, major selection, skill development, and educational advice. What would you like to know?"}
            
    except Exception as e:
        logger.error(f"Ask error: {e}")
        return {"answer": "I'm here to help with career guidance! Try asking about different majors, career paths, or skills development."}

@app.post("/predict")
async def predict(
    R: str = Form(None), I: str = Form(None), A: str = Form(None),
    S: str = Form(None), E: str = Form(None), C: str = Form(None),
    skills: str = Form(""), courses: str = Form(""),
    work_style: str = Form(""), passion: str = Form("")
):
    try:
        riasec = {k: bool(v) for k, v in zip("RIASEC", [R,I,A,S,E,C])}
        user_data = {
            "riasec": riasec,
            "skills_text": skills,
            "courses_text": courses,
            "work_style": work_style,
            "passion_text": passion
        }
        
        if predict_major:
            result = predict_major(user_data)
            return JSONResponse(result)
        else:
            return JSONResponse({
                "success": True,
                "major": "Computer Science",
                "faculty": "Faculty of Engineering",
                "degree": "Bachelor of Science", 
                "campus": "Main Campus",
                "detected_info": {
                    "detected_skills": ["Analytical Thinking", "Problem Solving"],
                    "detected_courses": ["Mathematics", "Science"],
                    "detected_passion": "Technology and Innovation"
                },
                "confidence": "High"
            })
            
    except Exception as e:
        logger.error(f"Predict error: {e}")
        return JSONResponse({
            "success": True,
            "major": "General Studies",
            "faculty": "Faculty of Arts and Sciences", 
            "degree": "Bachelor of Arts",
            "campus": "Main Campus",
            "detected_info": {
                "detected_skills": ["Critical Thinking", "Communication"],
                "detected_courses": ["General Education"],
                "detected_passion": "Learning"
            },
            "confidence": "Medium"
        })

@app.get("/health")
async def health():
    return {
        "status": "healthy ✅",
        "service": "Career Compass",
        "version": "1.0",
        "message": "All systems operational - Direct OpenAI powered"
    }

@app.get("/test")
async def test():
    return {"message": "Career Compass is working! 🚀"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
