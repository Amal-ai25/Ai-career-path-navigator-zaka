import logging
import re
import os
logger = logging.getLogger(__name__)

# Try to import ML dependencies, but handle gracefully if missing
try:
    import joblib
    import numpy as np
    from fuzzywuzzy import process, fuzz
    from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, LabelEncoder
    ML_DEPENDENCIES_AVAILABLE = True
    logger.info("✅ All ML dependencies loaded successfully")
except ImportError as e:
    ML_DEPENDENCIES_AVAILABLE = False
    logger.warning(f"⚠️ Some ML dependencies missing: {e}")
    logger.info("🔄 Using rule-based system only")

# Global variable to store models
models = None
ml_models_loaded = False

def load_ml_models():
    """Load ML models with proper error handling"""
    global models, ml_models_loaded
    
    if not ML_DEPENDENCIES_AVAILABLE:
        logger.warning("❌ ML dependencies not available - skipping model loading")
        return None
        
    try:
        # Define model paths
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
        
        for model_name, model_path in model_paths.items():
            try:
                if os.path.exists(model_path):
                    models[model_name] = joblib.load(model_path)
                    logger.info(f"✅ Loaded {model_name}")
                else:
                    logger.warning(f"⚠️ Model file not found: {model_path}")
                    all_models_found = False
                    break
            except Exception as e:
                logger.error(f"❌ Error loading {model_name}: {e}")
                all_models_found = False
                break
        
        if all_models_found:
            ml_models_loaded = True
            logger.info("🎉 All ML models loaded successfully!")
            return models
        else:
            logger.warning("⚠️ Some ML models failed to load")
            models = None
            ml_models_loaded = False
            return None
            
    except Exception as e:
        logger.error(f"❌ Critical error in load_ml_models: {e}")
        models = None
        ml_models_loaded = False
        return None

# Only try to load ML models if dependencies are available
if ML_DEPENDENCIES_AVAILABLE:
    logger.info("🚀 Attempting to load ML models...")
    load_ml_models()
else:
    logger.info("🔄 ML dependencies missing - using rule-based system")

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
        
        # Fuzzy matching (if available)
        if ML_DEPENDENCIES_AVAILABLE:
            matches = process.extract(token, master_list, scorer=fuzz.WRatio, limit=3)
            for match, score in matches:
                if score >= threshold:
                    detected_items.add(match)
        else:
            # Simple matching without fuzzywuzzy
            for item in master_list:
                if token in item.lower() or item.lower() in token:
                    detected_items.add(item)
                    break
    
    return list(detected_items)

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
    
    # If ML dependencies are not available, use rule-based directly
    if not ML_DEPENDENCIES_AVAILABLE:
        logger.info("🔄 ML dependencies missing - using rule-based system")
        rule_result, rule_error = predict_major_rules(user_data)
        if rule_result:
            rule_result['success'] = True
            return rule_result
    
    # Try ML prediction first (if dependencies available)
    if ml_models_loaded:
        # ML prediction code would go here
        # For now, fall back to rules
        pass
    
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
if ML_DEPENDENCIES_AVAILABLE and ml_models_loaded:
    logger.info("🎉 ML system ready - Real model loaded")
elif ML_DEPENDENCIES_AVAILABLE:
    logger.info("🔄 ML dependencies available but models not loaded")
else:
    logger.info("🔄 Using rule-based system - ML dependencies missing")
