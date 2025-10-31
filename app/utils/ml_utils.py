import re
import logging
import os
import sys

logger = logging.getLogger(__name__)

# Add models directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

ML_DEPENDENCIES_AVAILABLE = False
models = None
ml_models_loaded = False

logger.info("🔧 CHECKING ML DEPENDENCIES FOR NUMPY 2.0.2 COMPATIBILITY...")

try:
    import joblib
    logger.info(f"✅ joblib available (version: {joblib.__version__})")
except ImportError as e:
    logger.error(f"❌ joblib missing: {e}")

try:
    import numpy as np
    logger.info(f"✅ numpy available (version: {np.__version__})")
    # Test numpy 2.0.2 functionality
    test_array = np.array([1, 2, 3])
    logger.info(f"✅ Numpy 2.0.2 test passed: {test_array.shape}")
except ImportError as e:
    logger.error(f"❌ numpy missing: {e}")
except Exception as e:
    logger.error(f"❌ numpy test failed: {e}")

try:
    from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier
    logger.info(f"✅ scikit-learn available (version: sklearn.__version__)")
except ImportError as e:
    logger.error(f"❌ scikit-learn missing: {e}")

try:
    from fuzzywuzzy import process, fuzz
    logger.info("✅ fuzzywuzzy available")
except ImportError as e:
    logger.error(f"❌ fuzzywuzzy missing: {e}")

# Check if all are available
try:
    import joblib
    import numpy as np
    from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, LabelEncoder
    from fuzzywuzzy import process, fuzz
    ML_DEPENDENCIES_AVAILABLE = True
    logger.info("🎉 ALL ML DEPENDENCIES ARE AVAILABLE WITH NUMPY 2.0.2!")
except ImportError as e:
    ML_DEPENDENCIES_AVAILABLE = False
    logger.error(f"❌ Some ML dependencies missing: {e}")

def load_ml_models():
    """Load your Colab-trained ML models with numpy 2.0.2"""
    global models, ml_models_loaded
    
    if not ML_DEPENDENCIES_AVAILABLE:
        logger.error("❌ Cannot load models - dependencies missing")
        return None
        
    try:
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
        logger.info("🚀 LOADING ML MODELS WITH NUMPY 2.0.2...")
        
        for model_name, model_path in model_files.items():
            try:
                logger.info(f"🔄 Loading {model_name}...")
                
                if not os.path.exists(model_path):
                    logger.error(f"❌ Model file not found: {model_path}")
                    return None
                
                # Load with numpy 2.0.2 compatibility
                models[model_name] = joblib.load(model_path)
                logger.info(f"✅ Successfully loaded {model_name} with numpy 2.0.2")
                
            except Exception as e:
                logger.error(f"❌ ERROR loading {model_name}: {str(e)}")
                return None
        
        if len(models) == len(model_files):
            ml_models_loaded = True
            logger.info("🎉 ALL ML MODELS LOADED SUCCESSFULLY WITH NUMPY 2.0.2!")
            
            # Log detailed model information
            logger.info(f"📊 MODEL INFORMATION:")
            logger.info(f"   - Major classes: {list(models['le_major'].classes_)}")
            logger.info(f"   - Faculty classes: {list(models['le_faculty'].classes_)}")
            logger.info(f"   - Skills count: {len(models['all_skills'])}")
            logger.info(f"   - Courses count: {len(models['all_courses'])}")
            logger.info(f"   - Model type: {type(models['model'])}")
            
            return models
        else:
            logger.error(f"❌ Only loaded {len(models)}/{len(model_files)} models")
            return None
            
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR LOADING MODELS: {str(e)}")
        import traceback
        logger.error(f"💡 Full traceback: {traceback.format_exc()}")
        return None

# Try to load models on startup
if ML_DEPENDENCIES_AVAILABLE:
    logger.info("🚀 ATTEMPTING TO LOAD REAL ML MODELS WITH NUMPY 2.0.2...")
    models = load_ml_models()

def process_user_text_input(user_input, master_list, threshold=70):
    """
    Process user text input using fuzzy matching (from your Colab code)
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

def predict_with_real_ml(user_data):
    """USE YOUR ACTUAL COLAB-TRAINED ML MODEL WITH NUMPY 2.0.2"""
    if not ml_models_loaded or not models:
        return None, "ML models not loaded"
    
    try:
        logger.info("🧠 USING YOUR REAL ML MODEL (TRAINED WITH NUMPY 2.0.2)!")
        
        # Process RIASEC scores (from your Colab code)
        riasec_order = ['R', 'I', 'A', 'S', 'E', 'C']
        X_riasec = np.array([[user_data['riasec'].get(col, 0) for col in riasec_order]])
        logger.info(f"📊 RIASEC features: {X_riasec[0]}")
        
        # Process Skills with fuzzy matching
        skills_text = user_data.get('skills_text', '')
        detected_skills = process_user_text_input(skills_text, models['all_skills'])
        logger.info(f"🔧 Detected skills: {detected_skills}")
        X_skills = models['mlb_skills'].transform([detected_skills])
        
        # Process Courses with fuzzy matching
        courses_text = user_data.get('courses_text', '')
        detected_courses = process_user_text_input(courses_text, models['all_courses'])
        logger.info(f"📚 Detected courses: {detected_courses}")
        X_courses = models['mlb_courses'].transform([detected_courses])
        
        # Process Work Style
        work_style = user_data.get('work_style', '')
        if work_style not in models['all_work_styles']:
            work_style = models['all_work_styles'][0] if models['all_work_styles'] else 'Team-Oriented'
        X_work_style = models['ohe_work_style'].transform([[work_style]])
        logger.info(f"💼 Work style: {work_style}")
        
        # Process Passion with fuzzy matching
        passion_text = user_data.get('passion_text', '')
        detected_passions = process_user_text_input(passion_text, models['all_passions'])
        passion = detected_passions[0] if detected_passions else (models['all_passions'][0] if models['all_passions'] else 'Technology')
        X_passion = models['ohe_passion'].transform([[passion]])
        logger.info(f"❤️ Passion: {passion}")
        
        # Combine all features (from your Colab code)
        X_user = np.hstack([X_riasec, X_skills, X_courses, X_work_style, X_passion])
        logger.info(f"📈 Combined features shape: {X_user.shape}")
        
        # Make prediction using your actual trained model
        logger.info("🎯 Making prediction with your ML model...")
        prediction = models['model'].predict(X_user)
        logger.info(f"🎯 Raw prediction: {prediction[0]}")
        
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
            'method': 'Real ML Model (Colab-trained with numpy 2.0.2)'
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
    MAIN PREDICTION FUNCTION - USES REAL ML MODEL
    """
    logger.info("🤖 STARTING MAJOR PREDICTION WITH REAL ML MODEL")
    logger.info(f"📝 User input - Skills: '{user_data.get('skills_text', '')}', Passion: '{user_data.get('passion_text', '')}'")
    
    # ALWAYS TRY REAL ML MODEL FIRST
    if ml_models_loaded:
        logger.info("🚀 USING REAL ML MODEL FOR PREDICTION")
        ml_result, ml_error = predict_with_real_ml(user_data)
        if ml_result:
            ml_result['success'] = True
            logger.info("🎉 SUCCESS: Used real ML model for prediction")
            return ml_result
        else:
            logger.error(f"❌ Real ML model failed: {ml_error}")
    
    # If ML fails completely
    logger.error("❌ ML PREDICTION COMPLETELY FAILED")
    return {
        'major': "System Error",
        'faculty': "ML Model Failed", 
        'degree': "Please Try Again",
        'campus': "Support",
        'detected_info': {
            'detected_skills': ["Model loading issue"],
            'detected_courses': ["Check deployment logs"],
            'detected_passion': "Technical issue"
        },
        'success': False,
        'confidence': "Error",
        'method': 'ML Model Failed to Load',
        'error': 'Real ML model could not be loaded'
    }

# Log final status
if ml_models_loaded:
    logger.info("🎉 🎉 🎉 SYSTEM READY - REAL ML MODELS LOADED SUCCESSFULLY!")
    logger.info("🎯 All predictions will use your trained Colab ML model")
else:
    logger.error("❌ ❌ ❌ ML MODELS FAILED TO LOAD")
    logger.error("💡 Issue: numpy version compatibility")
    logger.error("💡 Colab used: numpy 2.0.2, scikit-learn 1.6.1, joblib 1.5.2")
    logger.error("💡 Make sure requirements.txt has exact matching versions")
