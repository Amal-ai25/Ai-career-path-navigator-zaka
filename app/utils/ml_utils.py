import pandas as pd
import numpy as np
import joblib
import re
from fuzzywuzzy import process, fuzz
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, LabelEncoder
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

def load_models():
    """Load all ML models and encoders from nested models/models/ directory"""
    global model, mlb_skills, mlb_courses, ohe_work_style, ohe_passion
    global le_major, le_faculty, le_degree, le_campus
    global all_skills, all_courses, all_passions, all_work_styles
    
    try:
        # Models are in nested models/models/ directory 
        models_dir = "models/models"
        
        logger.info(f"📁 Loading models from: {os.path.abspath(models_dir)}")
        
        # Check if models directory exists
        if not os.path.exists(models_dir):
            logger.error(f"❌ Models directory not found: {models_dir}")
            logger.info(f"📂 Current directory: {os.getcwd()}")
            logger.info(f"📂 Directory contents: {os.listdir('.')}")
            if os.path.exists('models'):
                logger.info(f"📂 models/ contents: {os.listdir('models')}")
            return False
        
        # List available model files
        model_files = os.listdir(models_dir)
        logger.info(f"📄 Available model files: {model_files}")
        
        # Required model files
        required_files = [
            "major_recommendation_model.pkl",
            "mlb_skills.pkl", "mlb_courses.pkl",
            "ohe_work_style.pkl", "ohe_passion.pkl", 
            "le_major.pkl", "le_faculty.pkl", "le_degree.pkl", "le_campus.pkl",
            "master_skills.pkl", "master_courses.pkl", 
            "master_passions.pkl", "master_work_styles.pkl"
        ]
        
        # Check if all required files exist
        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(models_dir, file)):
                missing_files.append(file)
        
        if missing_files:
            logger.error(f"❌ Missing model files: {missing_files}")
            return False
        
        # Load the main model
        model_path = os.path.join(models_dir, "major_recommendation_model.pkl")
        model = joblib.load(model_path)
        logger.info("✅ ML model loaded successfully")
        
        # Load all encoders and master lists
        encoders_to_load = {
            'mlb_skills': 'mlb_skills.pkl',
            'mlb_courses': 'mlb_courses.pkl', 
            'ohe_work_style': 'ohe_work_style.pkl',
            'ohe_passion': 'ohe_passion.pkl',
            'le_major': 'le_major.pkl',
            'le_faculty': 'le_faculty.pkl',
            'le_degree': 'le_degree.pkl',
            'le_campus': 'le_campus.pkl',
            'master_skills': 'master_skills.pkl',
            'master_courses': 'master_courses.pkl',
            'master_passions': 'master_passions.pkl',
            'master_work_styles': 'master_work_styles.pkl'
        }
        
        for var_name, file_name in encoders_to_load.items():
            file_path = os.path.join(models_dir, file_name)
            try:
                loaded_obj = joblib.load(file_path)
                globals()[var_name] = loaded_obj
                logger.info(f"✅ Loaded {var_name} from {file_name}")
            except Exception as e:
                logger.error(f"❌ Error loading {file_name}: {e}")
                return False
        
        logger.info("🎉 All ML components loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading ML models: {e}")
        import traceback
        logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
        return False

def process_user_text_input(user_input, master_list, threshold=70):
    """
    Process user text input using fuzzy matching (from your Colab)
    """
    if not user_input or not isinstance(user_input, str):
        return []

    detected_items = []
    # Try to match the whole input first
    best_match, score = process.extractOne(user_input, master_list, scorer=fuzz.partial_ratio)
    if score >= threshold:
        detected_items.append(best_match)

    # Also try to match individual words
    words = re.findall(r'\b\w+\b', user_input.lower())
    for word in words:
        if len(word) > 3:  # Only consider words longer than 3 characters
            best_match, score = process.extractOne(word, master_list, scorer=fuzz.partial_ratio)
            if score >= threshold:
                detected_items.append(best_match)

    return list(set(detected_items))  # Remove duplicates

def prepare_user_input(user_data):
    """
    Prepare user input for prediction (from your Colab)
    user_data should be a dictionary with:
    - riasec: dict with R,I,A,S,E,C as keys and 0/1 as values
    - skills_text: string of user skills
    - courses_text: string of user courses
    - work_style: string of selected work style
    - passion_text: string of user passion
    """
    global all_skills, all_courses, all_passions, all_work_styles
    global mlb_skills, mlb_courses, ohe_work_style, ohe_passion
    
    # Process RIASEC
    riasec_order = ['R', 'I', 'A', 'S', 'E', 'C']
    X_riasec = np.array([[user_data['riasec'].get(col, 0) for col in riasec_order]])

    # Process Skills with NLP
    detected_skills = process_user_text_input(user_data['skills_text'], all_skills)
    X_skills = mlb_skills.transform([detected_skills])

    # Process Courses with NLP
    detected_courses = process_user_text_input(user_data['courses_text'], all_courses)
    X_courses = mlb_courses.transform([detected_courses])

    # Process Work Style
    # If user's work style isn't found, use the most common one
    work_style = user_data['work_style']
    if work_style not in all_work_styles:
        work_style = all_work_styles[0]  # Use first available
    X_work_style = ohe_work_style.transform([[work_style]])

    # Process Passion with NLP
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
    """
    Predict major based on user input (from your Colab)
    """
    global model, le_major, le_faculty, le_degree, le_campus
    
    if model is None:
        return {"error": "ML model not loaded. Please try again later.", "success": False}
    
    try:
        # Prepare user input
        X_user, detected_info = prepare_user_input(user_data)

        # Make prediction
        prediction = model.predict(X_user)

        # Decode predictions
        result = {
            'major': le_major.inverse_transform([prediction[0][0]])[0],
            'faculty': le_faculty.inverse_transform([prediction[0][1]])[0],
            'degree': le_degree.inverse_transform([prediction[0][2]])[0],
            'campus': le_campus.inverse_transform([prediction[0][3]])[0],
            'detected_info': detected_info,
            'success': True
        }

        # Get probabilities for top recommendations
        if hasattr(model, 'predict_proba'):
            try:
                probas = [estimator.predict_proba(X_user)[0] for estimator in model.estimators_]
                major_probas = list(zip(le_major.classes_, probas[0]))
                major_probas.sort(key=lambda x: x[1], reverse=True)
                result['top_recommendations'] = [
                    {'major': major, 'confidence': float(confidence)} 
                    for major, confidence in major_probas[:3]  # Top 3 majors
                ]
            except Exception as e:
                logger.warning(f"Could not get probabilities: {e}")
                result['top_recommendations'] = []

        return result
        
    except Exception as e:
        logger.error(f"❌ Error in prediction: {e}")
        return {"error": f"Prediction failed: {str(e)}", "success": False}

# Load models when module is imported
logger.info("🔄 Attempting to load ML models from models/models/...")
load_models()
