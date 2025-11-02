import re
import joblib
import numpy as np
import os
import logging
from fuzzywuzzy import process, fuzz
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, LabelEncoder
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_model_path(filename):
    """Get correct model path whether running from app/ or project root"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try different possible model locations
    possible_paths = [
        os.path.join(current_dir, '..', '..', 'models', 'models', filename),  # From app/utils/
        os.path.join(current_dir, '..', 'models', 'models', filename),        # From app/
        os.path.join('models', 'models', filename),                           # From project root
        os.path.join('..', 'models', 'models', filename),                     # From app/utils/ with relative
        filename                                                              # Direct path
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"✅ Found model file: {path}")
            return path
        else:
            logger.debug(f"❌ Model not found at: {path}")
    
    # If no file found, return the most likely path and log error
    error_msg = f"Model file {filename} not found in any location. Checked: {possible_paths}"
    logger.error(error_msg)
    raise FileNotFoundError(error_msg)

# Load all models and encoders
def load_models():
    models = {}
    try:
        logger.info("🚀 Loading ML models...")
        
        models['model'] = joblib.load(get_model_path('major_recommendation_model.pkl'))
        models['mlb_skills'] = joblib.load(get_model_path('mlb_skills.pkl'))
        models['mlb_courses'] = joblib.load(get_model_path('mlb_courses.pkl'))
        models['ohe_work_style'] = joblib.load(get_model_path('ohe_work_style.pkl'))
        models['ohe_passion'] = joblib.load(get_model_path('ohe_passion.pkl'))
        models['le_major'] = joblib.load(get_model_path('le_major.pkl'))
        models['le_faculty'] = joblib.load(get_model_path('le_faculty.pkl'))
        models['le_degree'] = joblib.load(get_model_path('le_degree.pkl'))
        models['le_campus'] = joblib.load(get_model_path('le_campus.pkl'))
        
        # Load master lists
        models['all_skills'] = joblib.load(get_model_path('master_skills.pkl'))
        models['all_courses'] = joblib.load(get_model_path('master_courses.pkl'))
        models['all_passions'] = joblib.load(get_model_path('master_passions.pkl'))
        models['all_work_styles'] = joblib.load(get_model_path('master_work_styles.pkl'))
        
        # Add common variations
        additional_skills = [
            'Power BI', 'PowerBI', 'Data Analysis', 'Data Analytics', 'Business Intelligence',
            'AI', 'Artificial Intelligence', 'Machine Learning', 'Deep Learning',
            'Programming', 'Coding', 'Software Development', 'Web Development',
            'Data Science', 'Big Data', 'Cloud Computing', 'Cybersecurity'
        ]
        
        additional_courses = [
            'Mathematics', 'Math', 'Advanced Mathematics', 'Applied Mathematics',
            'AI', 'Artificial Intelligence', 'Machine Learning', 'Data Science',
            'Computer Science', 'Programming Fundamentals', 'Data Structures'
        ]
        
        additional_passions = [
            'AI', 'Artificial Intelligence', 'Machine Learning', 'Technology',
            'Data Science', 'Programming', 'Computer Science', 'Innovation'
        ]
        
        models['all_skills'].extend(additional_skills)
        models['all_courses'].extend(additional_courses)
        models['all_passions'].extend(additional_passions)
        
        # Remove duplicates
        models['all_skills'] = list(set(models['all_skills']))
        models['all_courses'] = list(set(models['all_courses']))
        models['all_passions'] = list(set(models['all_passions']))
        
        logger.info(f"✅ Loaded {len(models['all_skills'])} skills, {len(models['all_courses'])} courses, {len(models['all_passions'])} passions")
        logger.info(f"✅ Major classes: {list(models['le_major'].classes_)}")
        logger.info(f"✅ Faculty classes: {list(models['le_faculty'].classes_)}")
        
        return models
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        return None

# Load models on import
models = load_models()

def process_user_text_input(user_input, master_list, input_type="skills"):
    """
    Enhanced text processing with better handling of comma-separated values
    """
    if not user_input or not isinstance(user_input, str):
        return []
    
    detected_items = set()
    
    logger.info(f"🔍 Processing {input_type} input: '{user_input}'")
    
    # Clean and normalize the input
    user_input = user_input.strip()
    
    # Handle comma-separated values more intelligently
    tokens = []
    if ',' in user_input:
        # Split by commas but be careful with spaces
        parts = [part.strip() for part in user_input.split(',')]
        tokens.extend(parts)
    else:
        # Also split by spaces for single entries
        tokens = [user_input]
    
    # Remove empty tokens
    tokens = [token for token in tokens if token and len(token) > 1]
    
    logger.info(f"   Tokens extracted: {tokens}")
    
    for token in tokens:
        token = token.strip()
        if not token:
            continue
            
        logger.info(f"   🔎 Matching token: '{token}'")
        
        # Strategy 1: Exact match (case-insensitive)
        exact_matches = [item for item in master_list if token.lower() == item.lower()]
        if exact_matches:
            detected_items.update(exact_matches)
            logger.info(f"      ✅ Exact match: {exact_matches}")
            continue
        
        # Strategy 2: Partial match
        partial_matches = [item for item in master_list if token.lower() in item.lower() or item.lower() in token.lower()]
        if partial_matches:
            # Take the best partial match (longest or most specific)
            best_match = max(partial_matches, key=len)
            detected_items.add(best_match)
            logger.info(f"      ✅ Partial match: {best_match}")
            continue
        
        # Strategy 3: Fuzzy matching with multiple approaches
        # Try WRatio first (balanced approach)
        matches = process.extract(token, master_list, scorer=fuzz.WRatio, limit=10)
        for match, score in matches:
            if score >= 75:  # Good balance for spelling errors
                detected_items.add(match)
                logger.info(f"      ✅ Fuzzy match: {match} (score: {score})")
                break  # Take the best fuzzy match for this token
    
    final_items = list(detected_items)
    logger.info(f"🎯 Final {input_type}: {final_items}")
    return final_items

def prepare_user_input(user_data):
    """
    Prepare user input for prediction with extensive debugging
    """
    if not models:
        logger.error("❌ Models not loaded!")
        return None, "Models not loaded"
    
    logger.info(f"📥 Received user data: {user_data}")
    
    # Process RIASEC
    riasec_order = ['R', 'I', 'A', 'S', 'E', 'C']
    riasec_values = [1 if user_data['riasec'].get(col, False) else 0 for col in riasec_order]
    X_riasec = np.array([riasec_values])
    logger.info(f"🎭 RIASEC values: {dict(zip(riasec_order, riasec_values))}")
    
    # Process Skills
    skills_text = user_data.get('skills_text', '')
    detected_skills = process_user_text_input(skills_text, models['all_skills'], "skills")
    
    # If no skills detected but we have text, try to extract individual words
    if not detected_skills and skills_text:
        logger.info("🔄 Trying alternative skill extraction...")
        # Extract individual words and try to match them
        words = re.findall(r'\b\w+\b', skills_text.lower())
        for word in words:
            if len(word) > 3:  # Only consider words longer than 3 characters
                matches = process.extract(word, models['all_skills'], scorer=fuzz.partial_ratio, limit=3)
                for match, score in matches:
                    if score >= 80:
                        detected_skills.append(match)
                        break
        detected_skills = list(set(detected_skills))
        logger.info(f"🔄 Alternative skills detected: {detected_skills}")
    
    X_skills = models['mlb_skills'].transform([detected_skills])
    logger.info(f"🔧 Skills features shape: {X_skills.shape}")
    
    # Process Courses
    courses_text = user_data.get('courses_text', '')
    detected_courses = process_user_text_input(courses_text, models['all_courses'], "courses")
    X_courses = models['mlb_courses'].transform([detected_courses])
    logger.info(f"🔧 Courses features shape: {X_courses.shape}")
    
    # Process Work Style
    work_style = user_data.get('work_style', '')
    logger.info(f"💼 Work style: {work_style}")
    if work_style not in models['all_work_styles']:
        work_style = models['all_work_styles'][0] if models['all_work_styles'] else 'Office/Data'
        logger.info(f"🔄 Using default work style: {work_style}")
    X_work_style = models['ohe_work_style'].transform([[work_style]])
    logger.info(f"🔧 Work style features shape: {X_work_style.shape}")
    
    # Process Passion
    passion_text = user_data.get('passion_text', '')
    detected_passions = process_user_text_input(passion_text, models['all_passions'], "passions")
    passion = detected_passions[0] if detected_passions else (models['all_passions'][0] if models['all_passions'] else 'Technology')
    logger.info(f"❤️ Passion: {passion}")
    X_passion = models['ohe_passion'].transform([[passion]])
    logger.info(f"🔧 Passion features shape: {X_passion.shape}")
    
    # Combine all features
    X_user = np.hstack([X_riasec, X_skills, X_courses, X_work_style, X_passion])
    logger.info(f"🎯 Final feature vector shape: {X_user.shape}")
    
    detected_info = {
        'detected_skills': detected_skills,
        'detected_courses': detected_courses,
        'detected_passion': passion
    }
    
    logger.info(f"📊 Detected info: {detected_info}")
    
    return X_user, detected_info

def predict_major(user_data):
    """
    Predict major based on user input
    """
    if not models:
        return {"error": "Models not loaded. Please train the model first."}
    
    logger.info("🎯 Starting prediction...")
    
    # Prepare user input
    X_user, detected_info = prepare_user_input(user_data)
    
    if X_user is None:
        return {"error": "Failed to prepare user input"}
    
    # Make prediction
    try:
        prediction = models['model'].predict(X_user)
        logger.info(f"🤖 Model prediction: {prediction}")
        
        # Decode predictions
        result = {
            'major': models['le_major'].inverse_transform([prediction[0][0]])[0],
            'faculty': models['le_faculty'].inverse_transform([prediction[0][1]])[0],
            'degree': models['le_degree'].inverse_transform([prediction[0][2]])[0],
            'campus': models['le_campus'].inverse_transform([prediction[0][3]])[0],
            'detected_info': detected_info,
            'success': True
        }
        
        logger.info(f"✅ Prediction result: {result}")
        return result
        
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return {"error": error_msg, "success": False}

# Test function to verify everything works
def test_prediction():
    """Test the prediction system with sample data"""
    logger.info("🧪 Testing prediction system...")
    
    test_data = {
        'riasec': {'R': 0, 'I': 1, 'A': 0, 'S': 0, 'E': 0, 'C': 1},
        'skills_text': "python programming, data analysis, machine learning",
        'courses_text': "computer science, mathematics, statistics",
        'work_style': "Office/Data",
        'passion_text': "artificial intelligence and technology"
    }
    
    try:
        result = predict_major(test_data)
        logger.info(f"🧪 Test result: {result}")
        return result
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return {"error": str(e)}

# Run test if this file is executed directly
if __name__ == "__main__":
    logger.info("🔧 Running ML utils test...")
    test_prediction()
