import pandas as pd
import numpy as np
import joblib
import re
from fuzzywuzzy import process, fuzz
import logging
import os

logger = logging.getLogger(__name__)

# Global variables for models
model = None
mlb_skills = None
mlb_courses = None
ohe_work_style = None
ohe_passion = None
le_major = None
le_faculty = None
le_degree = None
le_campus = None
all_skills = None
all_courses = None
all_passions = None
all_work_styles = None

def safe_joblib_load(filepath):
    """Safely load joblib files with numpy compatibility"""
    try:
        return joblib.load(filepath)
    except Exception as e:
        logger.error(f"❌ Error loading {filepath}: {e}")
        # Try with specific encoding for numpy compatibility
        try:
            return joblib.load(filepath, encoding='latin1')
        except:
            raise e

def load_models():
    """Load all ML models and encoders with numpy compatibility fix"""
    global model, mlb_skills, mlb_courses, ohe_work_style, ohe_passion
    global le_major, le_faculty, le_degree, le_campus
    global all_skills, all_courses, all_passions, all_work_styles
    
    try:
        models_dir = "models/models"
        
        logger.info(f"📁 Loading models from: {os.path.abspath(models_dir)}")
        
        if not os.path.exists(models_dir):
            logger.error(f"❌ Models directory not found: {models_dir}")
            return False
        
        # Required files
        required_files = [
            "major_recommendation_model.pkl", "mlb_skills.pkl", "mlb_courses.pkl",
            "ohe_work_style.pkl", "ohe_passion.pkl", "le_major.pkl", "le_faculty.pkl",
            "le_degree.pkl", "le_campus.pkl", "master_skills.pkl", "master_courses.pkl",
            "master_passions.pkl", "master_work_styles.pkl"
        ]
        
        # Check files exist
        for file in required_files:
            if not os.path.exists(os.path.join(models_dir, file)):
                logger.error(f"❌ Missing model file: {file}")
                return False
        
        # Load models safely
        try:
            model = safe_joblib_load(os.path.join(models_dir, "major_recommendation_model.pkl"))
            logger.info("✅ ML model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading main model: {e}")
            return False
        
        # Load encoders
        encoders = {
            'mlb_skills': 'mlb_skills.pkl', 'mlb_courses': 'mlb_courses.pkl',
            'ohe_work_style': 'ohe_work_style.pkl', 'ohe_passion': 'ohe_passion.pkl',
            'le_major': 'le_major.pkl', 'le_faculty': 'le_faculty.pkl',
            'le_degree': 'le_degree.pkl', 'le_campus': 'le_campus.pkl',
            'master_skills': 'master_skills.pkl', 'master_courses': 'master_courses.pkl',
            'master_passions': 'master_passions.pkl', 'master_work_styles': 'master_work_styles.pkl'
        }
        
        for var_name, file_name in encoders.items():
            try:
                file_path = os.path.join(models_dir, file_name)
                globals()[var_name] = safe_joblib_load(file_path)
                logger.info(f"✅ Loaded {var_name}")
            except Exception as e:
                logger.error(f"❌ Error loading {file_name}: {e}")
                return False
        
        logger.info("🎉 All ML components loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading ML models: {e}")
        return False

def process_user_text_input(user_input, master_list, threshold=70):
    """Process user text input using fuzzy matching"""
    if not user_input or not isinstance(user_input, str):
        return []

    detected_items = []
    best_match, score = process.extractOne(user_input, master_list, scorer=fuzz.partial_ratio)
    if score >= threshold:
        detected_items.append(best_match)

    words = re.findall(r'\b\w+\b', user_input.lower())
    for word in words:
        if len(word) > 3:
            best_match, score = process.extractOne(word, master_list, scorer=fuzz.partial_ratio)
            if score >= threshold:
                detected_items.append(best_match)

    return list(set(detected_items))

def prepare_user_input(user_data):
    """Prepare user input for prediction"""
    global all_skills, all_courses, all_passions, all_work_styles
    global mlb_skills, mlb_courses, ohe_work_style, ohe_passion
    
    # Process RIASEC
    riasec_order = ['R', 'I', 'A', 'S', 'E', 'C']
    X_riasec = np.array([[user_data['riasec'].get(col, 0) for col in riasec_order]])

    # Process Skills
    detected_skills = process_user_text_input(user_data['skills_text'], all_skills)
    X_skills = mlb_skills.transform([detected_skills])

    # Process Courses
    detected_courses = process_user_text_input(user_data['courses_text'], all_courses)
    X_courses = mlb_courses.transform([detected_courses])

    # Process Work Style
    work_style = user_data['work_style']
    if work_style not in all_work_styles:
        work_style = all_work_styles[0]
    X_work_style = ohe_work_style.transform([[work_style]])

    # Process Passion
    detected_passion = process_user_text_input(user_data['passion_text'], all_passions)
    passion = detected_passion[0] if detected_passion else all_passions[0]
    X_passion = ohe_passion.transform([[passion]])

    # Combine all features
    X_user = np.hstack([X_riasec, X_skills, X_courses, X_work_style, X_passion])

    return X_user, {
        'detected_skills': detected_skills,
        'detected_courses': detected_courses,
        'detected_passion': passion
    }

def predict_major(user_data):
    """Predict major based on user input"""
    global model, le_major, le_faculty, le_degree, le_campus
    
    if model is None:
        return {"error": "ML model not loaded. Please try again later.", "success": False}
    
    try:
        X_user, detected_info = prepare_user_input(user_data)
        prediction = model.predict(X_user)

        result = {
            'major': le_major.inverse_transform([prediction[0][0]])[0],
            'faculty': le_faculty.inverse_transform([prediction[0][1]])[0],
            'degree': le_degree.inverse_transform([prediction[0][2]])[0],
            'campus': le_campus.inverse_transform([prediction[0][3]])[0],
            'detected_info': detected_info,
            'success': True
        }

        return result
        
    except Exception as e:
        logger.error(f"❌ Error in prediction: {e}")
        return {"error": f"Prediction failed: {str(e)}", "success": False}

# Load models on import
logger.info("🔄 Attempting to load ML models...")
load_models()
