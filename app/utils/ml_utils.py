import re
import logging
import os
import sys

logger = logging.getLogger(__name__)

# Add models directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Try to import with detailed error reporting
ML_DEPENDENCIES_AVAILABLE = False
models = None
ml_models_loaded = False

logger.info("🔧 CHECKING ML DEPENDENCIES...")

try:
    import joblib
    logger.info("✅ joblib available")
except ImportError as e:
    logger.error(f"❌ joblib missing: {e}")
    logger.error("💡 Run: pip install joblib==1.3.2")

try:
    import numpy as np
    logger.info(f"✅ numpy available (version: {np.__version__})")
except ImportError as e:
    logger.error(f"❌ numpy missing: {e}")
    logger.error("💡 Run: pip install numpy==1.24.3")

try:
    from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, LabelEncoder
    logger.info("✅ scikit-learn available")
except ImportError as e:
    logger.error(f"❌ scikit-learn missing: {e}")
    logger.error("💡 Run: pip install scikit-learn==1.3.2")

# Check if all are available
try:
    import joblib
    import numpy as np
    from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, LabelEncoder
    ML_DEPENDENCIES_AVAILABLE = True
    logger.info("🎉 ALL ML DEPENDENCIES ARE AVAILABLE!")
except ImportError:
    ML_DEPENDENCIES_AVAILABLE = False
    logger.error("❌ SOME ML DEPENDENCIES ARE MISSING!")

def load_ml_models():
    """Load your actual ML models from models/models/ folder"""
    global models, ml_models_loaded
    
    if not ML_DEPENDENCIES_AVAILABLE:
        logger.error("❌ CANNOT LOAD MODELS - ML DEPENDENCIES MISSING")
        logger.error("💡 Please install: joblib, numpy, scikit-learn")
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
        logger.info("🚀 LOADING YOUR REAL ML MODELS...")
        
        # Load each model file
        for model_name, model_path in model_files.items():
            try:
                logger.info(f"🔄 Loading {model_name} from {model_path}...")
                
                # Check if file exists
                if not os.path.exists(model_path):
                    logger.error(f"❌ Model file not found: {model_path}")
                    logger.error("💡 Make sure your model files are in models/models/")
                    return None
                
                # Load the model
                models[model_name] = joblib.load(model_path)
                logger.info(f"✅ Successfully loaded {model_name}")
                
            except Exception as e:
                logger.error(f"❌ ERROR loading {model_name}: {str(e)}")
                logger.error(f"💡 File path: {os.path.abspath(model_path)}")
                return None
        
        # Verify all models loaded
        if len(models) == len(model_files):
            ml_models_loaded = True
            logger.info("🎉 ALL REAL ML MODELS LOADED SUCCESSFULLY!")
            logger.info(f"📊 Model info:")
            logger.info(f"   - Major classes: {list(models['le_major'].classes_)}")
            logger.info(f"   - Faculty classes: {list(models['le_faculty'].classes_)}")
            logger.info(f"   - Skills count: {len(models['all_skills'])}")
            logger.info(f"   - Courses count: {len(models['all_courses'])}")
            return models
        else:
            logger.error(f"❌ Only loaded {len(models)}/{len(model_files)} models")
            return None
            
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR loading ML models: {str(e)}")
        import traceback
        logger.error(f"💡 Full traceback: {traceback.format_exc()}")
        return None

# Try to load the real ML models on startup
if ML_DEPENDENCIES_AVAILABLE:
    logger.info("🚀 ATTEMPTING TO LOAD REAL ML MODELS ON STARTUP...")
    models = load_ml_models()
else:
    logger.error("🔄 CANNOT LOAD ML MODELS - DEPENDENCIES MISSING")

def enhanced_text_match(user_input, master_list):
    """Enhanced text matching for ML model input"""
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
            if len(common_words) >= 1:
                detected_items.add(item)
    
    return list(detected_items)[:3]

def predict_with_real_ml(user_data):
    """USE YOUR ACTUAL TRAINED ML MODEL FOR PREDICTION"""
    if not ml_models_loaded or not models:
        error_msg = "REAL ML MODELS NOT LOADED"
        logger.error(f"❌ {error_msg}")
        return None, error_msg
    
    try:
        logger.info("🧠 USING YOUR REAL ML MODEL FOR PREDICTION!")
        
        # Process RIASEC scores
        riasec_order = ['R', 'I', 'A', 'S', 'E', 'C']
        X_riasec = np.array([[user_data['riasec'].get(col, 0) for col in riasec_order]])
        logger.info(f"📊 RIASEC features: {X_riasec}")
        
        # Process skills using your actual MLB
        skills_text = user_data.get('skills_text', '')
        detected_skills = enhanced_text_match(skills_text, models['all_skills'])
        logger.info(f"🔧 Detected skills: {detected_skills}")
        X_skills = models['mlb_skills'].transform([detected_skills])
        logger.info(f"🔧 Skills features shape: {X_skills.shape}")
        
        # Process courses using your actual MLB
        courses_text = user_data.get('courses_text', '')
        detected_courses = enhanced_text_match(courses_text, models['all_courses'])
        logger.info(f"📚 Detected courses: {detected_courses}")
        X_courses = models['mlb_courses'].transform([detected_courses])
        logger.info(f"📚 Courses features shape: {X_courses.shape}")
        
        # Process work style using your actual OHE
        work_style = user_data.get('work_style', '')
        if work_style not in models['all_work_styles']:
            work_style = models['all_work_styles'][0] if models['all_work_styles'] else 'Team-Oriented'
        X_work_style = models['ohe_work_style'].transform([[work_style]])
        logger.info(f"💼 Work style: {work_style}")
        logger.info(f"💼 Work style features shape: {X_work_style.shape}")
        
        # Process passion using your actual OHE
        passion_text = user_data.get('passion_text', '')
        detected_passions = enhanced_text_match(passion_text, models['all_passions'])
        passion = detected_passions[0] if detected_passions else (models['all_passions'][0] if models['all_passions'] else 'Technology')
        X_passion = models['ohe_passion'].transform([[passion]])
        logger.info(f"❤️ Passion: {passion}")
        logger.info(f"❤️ Passion features shape: {X_passion.shape}")
        
        # Combine all features
        X_user = np.hstack([X_riasec, X_skills, X_courses, X_work_style, X_passion])
        logger.info(f"📈 Combined features shape: {X_user.shape}")
        
        # Make prediction using your actual trained model
        logger.info("🎯 Making prediction with your ML model...")
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
            'method': 'REAL ML MODEL'
        }
        
        logger.info(f"✅ REAL ML PREDICTION RESULT: {major} in {faculty}")
        return result, None
        
    except Exception as e:
        logger.error(f"❌ REAL ML PREDICTION FAILED: {str(e)}")
        import traceback
        logger.error(f"💡 Full error: {traceback.format_exc()}")
        return None, f"ML prediction error: {str(e)}"

def predict_major(user_data):
    """
    MAIN PREDICTION FUNCTION - FORCES REAL ML MODEL USAGE
    """
    logger.info("🤖 STARTING MAJOR PREDICTION")
    logger.info(f"📝 User data: RIASEC={user_data.get('riasec', {})}, Skills='{user_data.get('skills_text', '')}'")
    
    # ALWAYS TRY REAL ML MODEL FIRST
    if ML_DEPENDENCIES_AVAILABLE and ml_models_loaded:
        logger.info("🚀 USING REAL ML MODEL FOR PREDICTION")
        ml_result, ml_error = predict_with_real_ml(user_data)
        if ml_result:
            ml_result['success'] = True
            logger.info("🎉 SUCCESS: Used real ML model for prediction")
            return ml_result
        else:
            logger.error(f"❌ Real ML model failed: {ml_error}")
    
    # If ML fails, show clear error
    error_result = {
        'major': "SYSTEM ERROR",
        'faculty': "ML Model Not Loaded",
        'degree': "Please Check Dependencies",
        'campus': "Contact Support",
        'detected_info': {
            'detected_skills': ["ML dependencies missing"],
            'detected_courses': ["Install joblib, numpy, scikit-learn"],
            'detected_passion': "System configuration issue"
        },
        'success': False,
        'confidence': "Error",
        'method': 'ML MODEL FAILED - CHECK DEPENDENCIES',
        'error': 'ML dependencies not available. Please install: joblib, numpy, scikit-learn'
    }
    
    logger.error("❌ ML PREDICTION FAILED - Returning error result")
    return error_result

# Log final status
if ml_models_loaded:
    logger.info("🎉 SYSTEM READY - REAL ML MODELS ARE ACTIVE!")
    logger.info("🎯 Predictions will use your trained ML model")
else:
    logger.error("❌ SYSTEM NOT READY - ML MODELS FAILED TO LOAD")
    logger.error("💡 Check that all dependencies are installed")
    logger.error("💡 Required: joblib, numpy, scikit-learn")
