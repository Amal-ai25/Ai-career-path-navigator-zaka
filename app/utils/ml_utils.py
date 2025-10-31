import re
import logging
import os
import sys

logger = logging.getLogger(__name__)

# Try to import required ML dependencies
try:
    import joblib
    import numpy as np
    from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, LabelEncoder
    ML_DEPENDENCIES_AVAILABLE = True
    logger.info("✅ All ML dependencies available")
except ImportError as e:
    ML_DEPENDENCIES_AVAILABLE = False
    logger.error(f"❌ ML dependencies missing: {e}")

# Global variable to store your actual models
models = None
ml_models_loaded = False

def load_ml_models():
    """Load your actual ML models from models/models/ folder"""
    global models, ml_models_loaded
    
    if not ML_DEPENDENCIES_AVAILABLE:
        logger.error("❌ Cannot load models - ML dependencies not available")
        return None
        
    try:
        # Your actual model file paths
        model_files = {
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
        
        # Load each model file
        for model_name, model_path in model_files.items():
            try:
                if os.path.exists(model_path):
                    models[model_name] = joblib.load(model_path)
                    logger.info(f"✅ Loaded {model_name}")
                else:
                    logger.error(f"❌ Model file not found: {model_path}")
                    return None
            except Exception as e:
                logger.error(f"❌ Error loading {model_name}: {e}")
                return None
        
        ml_models_loaded = True
        logger.info("🎉 ALL REAL ML MODELS LOADED SUCCESSFULLY!")
        logger.info(f"   - Model type: {type(models['model'])}")
        logger.info(f"   - Skills: {len(models['all_skills'])} items")
        logger.info(f"   - Courses: {len(models['all_courses'])} items")
        logger.info(f"   - Majors: {list(models['le_major'].classes_)}")
        return models
            
    except Exception as e:
        logger.error(f"❌ Critical error loading ML models: {e}")
        return None

# Try to load the real ML models on startup
if ML_DEPENDENCIES_AVAILABLE:
    logger.info("🚀 LOADING YOUR REAL ML MODELS FROM models/models/...")
    models = load_ml_models()

def simple_text_match(user_input, master_list):
    """Simple text matching for skills/courses/passion"""
    if not user_input or not isinstance(user_input, str) or not master_list:
        return []
    
    detected_items = set()
    user_input_lower = user_input.lower()
    
    for item in master_list:
        if not isinstance(item, str):
            continue
            
        item_lower = item.lower()
        
        # Exact match or partial match
        if item_lower in user_input_lower or user_input_lower in item_lower:
            detected_items.add(item)
        else:
            # Word-based matching
            input_words = set(re.findall(r'\w+', user_input_lower))
            item_words = set(re.findall(r'\w+', item_lower))
            common_words = input_words.intersection(item_words)
            if len(common_words) >= 1:  # At least one common word
                detected_items.add(item)
    
    return list(detected_items)[:3]  # Limit to 3 best matches

def predict_with_real_ml(user_data):
    """Use your actual trained ML model for prediction"""
    if not ml_models_loaded or not models:
        return None, "Real ML models not loaded"
    
    try:
        logger.info("🧠 USING YOUR REAL ML MODEL FOR PREDICTION")
        
        # Process RIASEC scores
        riasec_order = ['R', 'I', 'A', 'S', 'E', 'C']
        X_riasec = np.array([[user_data['riasec'].get(col, 0) for col in riasec_order]])
        logger.info(f"📊 RIASEC features: {X_riasec}")
        
        # Process skills using your actual MLB
        skills_text = user_data.get('skills_text', '')
        detected_skills = simple_text_match(skills_text, models['all_skills'])
        logger.info(f"🔧 Detected skills: {detected_skills}")
        X_skills = models['mlb_skills'].transform([detected_skills])
        
        # Process courses using your actual MLB
        courses_text = user_data.get('courses_text', '')
        detected_courses = simple_text_match(courses_text, models['all_courses'])
        logger.info(f"📚 Detected courses: {detected_courses}")
        X_courses = models['mlb_courses'].transform([detected_courses])
        
        # Process work style using your actual OHE
        work_style = user_data.get('work_style', '')
        if work_style not in models['all_work_styles']:
            work_style = models['all_work_styles'][0] if models['all_work_styles'] else 'Team-Oriented'
        X_work_style = models['ohe_work_style'].transform([[work_style]])
        logger.info(f"💼 Work style: {work_style}")
        
        # Process passion using your actual OHE
        passion_text = user_data.get('passion_text', '')
        detected_passions = simple_text_match(passion_text, models['all_passions'])
        passion = detected_passions[0] if detected_passions else (models['all_passions'][0] if models['all_passions'] else 'Technology')
        X_passion = models['ohe_passion'].transform([[passion]])
        logger.info(f"❤️ Passion: {passion}")
        
        # Combine all features
        X_user = np.hstack([X_riasec, X_skills, X_courses, X_work_style, X_passion])
        logger.info(f"📈 Combined features shape: {X_user.shape}")
        
        # Make prediction using your actual trained model
        prediction = models['model'].predict(X_user)
        logger.info(f"🎯 Raw prediction: {prediction}")
        
        # Decode predictions using your actual label encoders
        major = models['le_major'].inverse_transform([prediction[0][0]])[0]
        faculty = models['le_faculty'].inverse_transform([prediction[0][1]])[0]
        degree = models['le_degree'].inverse_transform([prediction[0][2]])[0]
        campus = models['le_campus'].inverse_transform([prediction[0][3]])[0]
        
        result = {
            'major': major,
            'faculty': faculty,
            'degree': degree,
            'campus': campus,
            'detected_info': {
                'detected_skills': detected_skills,
                'detected_courses': detected_courses,
                'detected_passion': passion
            },
            'confidence': 'High',
            'method': 'Real ML Model'
        }
        
        logger.info(f"✅ REAL ML PREDICTION: {major} in {faculty}")
        return result, None
        
    except Exception as e:
        logger.error(f"❌ Real ML prediction failed: {e}")
        return None, f"ML prediction error: {str(e)}"

def predict_major_rules(user_data):
    """Rule-based fallback (only if ML fails)"""
    try:
        riasec = user_data.get('riasec', {})
        skills = user_data.get('skills_text', '').lower()
        passion = user_data.get('passion_text', '').lower()
        
        # Simple rule-based logic as fallback
        if riasec.get('I', 0) or any(word in skills for word in ['programming', 'python', 'java']):
            major = "Computer Science"
            faculty = "Faculty of Engineering"
        elif riasec.get('E', 0) or any(word in skills for word in ['business', 'management']):
            major = "Business Administration"
            faculty = "Faculty of Business"
        elif riasec.get('A', 0) or any(word in skills for word in ['design', 'art']):
            major = "Architecture" 
            faculty = "Faculty of Fine Arts"
        else:
            major = "General Studies"
            faculty = "Faculty of Arts and Sciences"
        
        return {
            'major': major,
            'faculty': faculty,
            'degree': "Bachelor of Science",
            'campus': "Main Campus",
            'detected_info': {
                'detected_skills': ["Based on your input"],
                'detected_courses': ["General"],
                'detected_passion': "Various"
            },
            'confidence': 'Medium',
            'method': 'Rule-Based Fallback'
        }, None
        
    except Exception as e:
        return None, f"Rule-based error: {str(e)}"

def predict_major(user_data):
    """
    Main prediction function - uses REAL ML model if available
    """
    logger.info("🤖 STARTING MAJOR PREDICTION")
    logger.info(f"📝 User data: {user_data}")
    
    # Try REAL ML model first
    if ml_models_loaded:
        ml_result, ml_error = predict_with_real_ml(user_data)
        if ml_result:
            ml_result['success'] = True
            logger.info("🎯 USING REAL ML MODEL PREDICTION")
            return ml_result
        else:
            logger.warning(f"🔄 Real ML failed: {ml_error}")
    
    # Fall back to rule-based
    rule_result, rule_error = predict_major_rules(user_data)
    if rule_result:
        rule_result['success'] = True
        logger.info("🔄 Using rule-based fallback")
        return rule_result
    
    # Ultimate fallback
    logger.error("❌ All prediction methods failed")
    return {
        'major': "Computer Science",
        'faculty': "Faculty of Science",
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

# Log final status
if ml_models_loaded:
    logger.info("🎉 SYSTEM READY - USING YOUR REAL ML MODELS!")
else:
    logger.info("🔄 SYSTEM READY - Using rule-based system (ML dependencies missing)")
