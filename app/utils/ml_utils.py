import re
import joblib
import numpy as np
import logging
from fuzzywuzzy import process, fuzz
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, LabelEncoder
import os

logger = logging.getLogger(__name__)

# Global variable to store models
models = None
ml_models_loaded = False

def load_ml_models():
    """Load ML models with proper error handling"""
    global models, ml_models_loaded
    
    try:
        # Define model paths - CORRECTED PATH with double models folder
        model_paths = {
            'model': 'models/models/major_recommendation_model.pkl',
            'mlb_skills': 'models/models/mlb_skills.pkl', 
            'mlb_courses': 'models/models/mlb_courses.pkl',
            'ohe_work_style': 'models/models/ohe_work_style.pkl',
            'ohe_passion': 'models/models/ohe_passion.pkl',
            'le_major': 'models/models/le_major.pkl',
            'le_faculty': 'models/models/le_faculty.pkl',
            'le_degree': 'models/models/le_degree.pkl',
            'le_campus': 'models/models/le_campus.pkl',
            'all_skills': 'models/models/master_skills.pkl',
            'all_courses': 'models/models/master_courses.pkl',
            'all_passions': 'models/models/master_passions.pkl',
            'all_work_styles': 'models/models/master_work_styles.pkl'
        }
        
        models = {}
        all_models_found = True
        
        # Try to load each model file
        for model_name, model_path in model_paths.items():
            try:
                # Check if file exists with absolute path
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(current_dir))
                absolute_path = os.path.join(project_root, model_path)
                
                if os.path.exists(absolute_path):
                    models[model_name] = joblib.load(absolute_path)
                    logger.info(f"✅ Loaded {model_name} from {absolute_path}")
                elif os.path.exists(model_path):
                    models[model_name] = joblib.load(model_path)
                    logger.info(f"✅ Loaded {model_name} from {model_path}")
                else:
                    logger.warning(f"⚠️ Model file not found: {model_path}")
                    logger.warning(f"⚠️ Also not found at: {absolute_path}")
                    all_models_found = False
                    break
            except Exception as e:
                logger.error(f"❌ Error loading {model_name} from {model_path}: {e}")
                all_models_found = False
                break
        
        if all_models_found:
            ml_models_loaded = True
            logger.info("🎉 All ML models loaded successfully!")
            return models
        else:
            logger.warning("⚠️ Some ML models failed to load - using rule-based fallback")
            models = None
            ml_models_loaded = False
            return None
            
    except Exception as e:
        logger.error(f"❌ Critical error in load_ml_models: {e}")
        models = None
        ml_models_loaded = False
        return None

# Try to load ML models on import
logger.info("🚀 Attempting to load ML models from models/models/ folder...")
load_ml_models()

def process_user_text_input(user_input, master_list, threshold=70):
    """Enhanced text processing with fuzzy matching"""
    if not user_input or not isinstance(user_input, str):
        return []
    
    detected_items = set()
    tokens = re.split(r'[,\n;]+', user_input.lower())
    
    for token in tokens:
        token = token.strip()
        if not token or len(token) < 2:
            continue
        
        # Exact match
        exact_matches = [item for item in master_list if token == item.lower()]
        if exact_matches:
            detected_items.update(exact_matches)
            continue
        
        # Partial match
        partial_matches = [item for item in master_list if token in item.lower() or item.lower() in token]
        if partial_matches:
            detected_items.update(partial_matches)
            continue
        
        # Fuzzy matching
        matches = process.extract(token, master_list, scorer=fuzz.WRatio, limit=3)
        for match, score in matches:
            if score >= threshold:
                detected_items.add(match)
    
    return list(detected_items)

def prepare_user_input_ml(user_data):
    """Prepare user input for ML prediction"""
    if not models:
        return None, "Models not loaded"
    
    try:
        # Process RIASEC
        riasec_order = ['R', 'I', 'A', 'S', 'E', 'C']
        X_riasec = np.array([[user_data['riasec'].get(col, 0) for col in riasec_order]])
        
        # Process Skills
        detected_skills = process_user_text_input(user_data['skills_text'], models['all_skills'], threshold=65)
        X_skills = models['mlb_skills'].transform([detected_skills])
        
        # Process Courses
        detected_courses = process_user_text_input(user_data['courses_text'], models['all_courses'], threshold=65)
        X_courses = models['mlb_courses'].transform([detected_courses])
        
        # Process Work Style
        work_style = user_data['work_style']
        if work_style not in models['all_work_styles']:
            work_style = models['all_work_styles'][0]
        X_work_style = models['ohe_work_style'].transform([[work_style]])
        
        # Process Passion
        detected_passions = process_user_text_input(user_data['passion_text'], models['all_passions'], threshold=65)
        passion = detected_passions[0] if detected_passions else models['all_passions'][0]
        X_passion = models['ohe_passion'].transform([[passion]])
        
        # Combine features
        X_user = np.hstack([X_riasec, X_skills, X_courses, X_work_style, X_passion])
        
        return X_user, {
            'detected_skills': detected_skills,
            'detected_courses': detected_courses,
            'detected_passion': passion
        }
        
    except Exception as e:
        logger.error(f"❌ Error preparing ML input: {e}")
        return None, f"Input preparation failed: {str(e)}"

def predict_major_ml(user_data):
    """Predict using real ML model"""
    if not ml_models_loaded or not models:
        return None, "ML models not available"
    
    try:
        X_user, detected_info = prepare_user_input_ml(user_data)
        
        if X_user is None:
            return None, "Failed to prepare input"
        
        # Make prediction
        prediction = models['model'].predict(X_user)
        
        # Decode predictions
        result = {
            'major': models['le_major'].inverse_transform([prediction[0][0]])[0],
            'faculty': models['le_faculty'].inverse_transform([prediction[0][1]])[0],
            'degree': models['le_degree'].inverse_transform([prediction[0][2]])[0],
            'campus': models['le_campus'].inverse_transform([prediction[0][3]])[0],
            'detected_info': detected_info,
            'confidence': 'High',
            'method': 'ML Model'
        }
        
        logger.info(f"✅ ML Prediction: {result['major']}")
        return result, None
        
    except Exception as e:
        logger.error(f"❌ ML prediction failed: {e}")
        return None, f"ML prediction error: {str(e)}"

def predict_major_rules(user_data):
    """Rule-based fallback prediction"""
    try:
        riasec = user_data.get('riasec', {})
        skills = user_data.get('skills_text', '').lower()
        courses = user_data.get('courses_text', '').lower()
        passion = user_data.get('passion_text', '').lower()
        work_style = user_data.get('work_style', '')
        
        # Smart recommendation logic
        if riasec.get('I', 0) or any(word in skills for word in ['programming', 'python', 'java', 'code', 'software']):
            major = "Computer Science"
            faculty = "Faculty of Engineering"
        elif riasec.get('E', 0) or any(word in skills for word in ['business', 'management', 'leadership', 'marketing']):
            major = "Business Administration" 
            faculty = "Faculty of Business"
        elif riasec.get('A', 0) or any(word in skills for word in ['design', 'art', 'creative', 'drawing']):
            major = "Architecture"
            faculty = "Faculty of Fine Arts"
        elif riasec.get('S', 0) or any(word in skills for word in ['teaching', 'helping', 'care', 'people']):
            major = "Psychology"
            faculty = "Faculty of Arts and Sciences"
        else:
            major = "General Studies"
            faculty = "Faculty of Arts and Sciences"
        
        # Detect skills
        detected_skills = []
        if any(word in skills for word in ['python', 'java', 'c++', 'programming']):
            detected_skills.append("Programming")
        if any(word in skills for word in ['design', 'creative', 'art']):
            detected_skills.append("Design")
        if any(word in skills for word in ['management', 'leadership']):
            detected_skills.append("Management")
        if any(word in skills for word in ['analysis', 'research', 'data']):
            detected_skills.append("Analytical Skills")
            
        if not detected_skills:
            detected_skills = ["Critical Thinking", "Problem Solving"]
        
        # Detect courses
        detected_courses = []
        if any(word in courses for word in ['computer', 'programming', 'coding']):
            detected_courses.append("Computer Science")
        if any(word in courses for word in ['business', 'economics']):
            detected_courses.append("Business")
        if any(word in courses for word in ['math', 'calculus', 'algebra']):
            detected_courses.append("Mathematics")
            
        if not detected_courses:
            detected_courses = ["General Education"]
        
        # Detect passion
        if any(word in passion for word in ['tech', 'computer', 'software', 'ai']):
            detected_passion = "Technology"
        elif any(word in passion for word in ['business', 'entrepreneur', 'startup']):
            detected_passion = "Business"
        elif any(word in passion for word in ['art', 'design', 'creative']):
            detected_passion = "Arts and Design"
        elif any(word in passion for word in ['help', 'people', 'community']):
            detected_passion = "Helping Others"
        else:
            detected_passion = "Learning and Development"
        
        result = {
            'major': major,
            'faculty': faculty,
            'degree': "Bachelor of Science" if "Engineering" in faculty else "Bachelor of Arts",
            'campus': "Main Campus",
            'detected_info': {
                'detected_skills': detected_skills,
                'detected_courses': detected_courses,
                'detected_passion': detected_passion
            },
            'confidence': "Medium",
            'method': 'Rule-Based'
        }
        
        logger.info(f"📊 Rule-based Prediction: {result['major']}")
        return result, None
        
    except Exception as e:
        logger.error(f"❌ Rule-based prediction failed: {e}")
        return None, f"Rule-based prediction error: {str(e)}"

def predict_major(user_data):
    """
    Hybrid prediction - tries ML first, falls back to rules
    """
    logger.info("🤖 Starting major prediction...")
    
    # Try ML prediction first
    if ml_models_loaded:
        ml_result, ml_error = predict_major_ml(user_data)
        if ml_result:
            ml_result['success'] = True
            logger.info("🎯 Using ML model prediction")
            return ml_result
        else:
            logger.warning(f"🔄 ML prediction failed: {ml_error}")
    
    # Fall back to rule-based system
    rule_result, rule_error = predict_major_rules(user_data)
    if rule_result:
        rule_result['success'] = True
        logger.info("🔄 Using rule-based prediction")
        return rule_result
    
    # Ultimate fallback
    logger.error("❌ Both ML and rule-based predictions failed")
    return {
        'major': "Computer Science",
        'faculty': "Faculty of Engineering", 
        'degree': "Bachelor of Science",
        'campus': "Main Campus",
        'detected_info': {
            'detected_skills': ["Analytical Thinking"],
            'detected_courses': ["General Courses"],
            'detected_passion': "Technology"
        },
        'success': True,
        'confidence': "Low",
        'method': 'Emergency Fallback'
    }

# Log system status
if ml_models_loaded:
    logger.info("🎉 ML system ready - Real model loaded from models/models/")
else:
    logger.info("🔄 ML system using rule-based fallback - models not found in models/models/")
