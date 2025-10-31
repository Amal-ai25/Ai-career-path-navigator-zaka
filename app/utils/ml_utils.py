import re
import logging
import os

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
        return models
            
    except Exception as e:
        logger.error(f"❌ Critical error loading ML models: {e}")
        return None

# Try to load the real ML models on startup
if ML_DEPENDENCIES_AVAILABLE:
    logger.info("🚀 LOADING YOUR REAL ML MODELS...")
    models = load_ml_models()

def enhanced_text_match(user_input, master_list):
    """Enhanced text matching without fuzzywuzzy"""
    if not user_input or not isinstance(user_input, str) or not master_list:
        return []
    
    detected_items = set()
    user_input_lower = user_input.lower()
    
    # Split input into words
    input_words = set(re.findall(r'\w+', user_input_lower))
    
    for item in master_list:
        if not isinstance(item, str):
            continue
            
        item_lower = item.lower()
        item_words = set(re.findall(r'\w+', item_lower))
        
        # Strategy 1: Exact match
        if item_lower in user_input_lower or user_input_lower in item_lower:
            detected_items.add(item)
            continue
            
        # Strategy 2: Multiple word matches
        common_words = input_words.intersection(item_words)
        if len(common_words) >= 2:  # At least 2 common words
            detected_items.add(item)
            continue
            
        # Strategy 3: Single important word match
        important_words = ['medical', 'lab', 'research', 'programming', 'business', 'design', 'art', 'teaching', 'helping']
        if any(word in input_words for word in important_words) and any(word in item_words for word in important_words):
            detected_items.add(item)
    
    return list(detected_items)[:3]

def predict_with_real_ml(user_data):
    """Use your actual trained ML model for prediction"""
    if not ml_models_loaded or not models:
        return None, "Real ML models not loaded"
    
    try:
        logger.info("🧠 USING YOUR REAL ML MODEL FOR PREDICTION")
        
        # Process RIASEC scores
        riasec_order = ['R', 'I', 'A', 'S', 'E', 'C']
        X_riasec = np.array([[user_data['riasec'].get(col, 0) for col in riasec_order]])
        logger.info(f"📊 RIASEC: {X_riasec}")
        
        # Process skills
        skills_text = user_data.get('skills_text', '')
        detected_skills = enhanced_text_match(skills_text, models['all_skills'])
        logger.info(f"🔧 Skills: {detected_skills}")
        X_skills = models['mlb_skills'].transform([detected_skills])
        
        # Process courses
        courses_text = user_data.get('courses_text', '')
        detected_courses = enhanced_text_match(courses_text, models['all_courses'])
        logger.info(f"📚 Courses: {detected_courses}")
        X_courses = models['mlb_courses'].transform([detected_courses])
        
        # Process work style
        work_style = user_data.get('work_style', '')
        if work_style not in models['all_work_styles']:
            work_style = models['all_work_styles'][0] if models['all_work_styles'] else 'Team-Oriented'
        X_work_style = models['ohe_work_style'].transform([[work_style]])
        logger.info(f"💼 Work style: {work_style}")
        
        # Process passion
        passion_text = user_data.get('passion_text', '')
        detected_passions = enhanced_text_match(passion_text, models['all_passions'])
        passion = detected_passions[0] if detected_passions else (models['all_passions'][0] if models['all_passions'] else 'Technology')
        X_passion = models['ohe_passion'].transform([[passion]])
        logger.info(f"❤️ Passion: {passion}")
        
        # Combine features and predict
        X_user = np.hstack([X_riasec, X_skills, X_courses, X_work_style, X_passion])
        prediction = models['model'].predict(X_user)
        
        # Decode predictions
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
        
        logger.info(f"✅ REAL ML PREDICTION: {major}")
        return result, None
        
    except Exception as e:
        logger.error(f"❌ Real ML prediction failed: {e}")
        return None, f"ML prediction error: {str(e)}"

def predict_major_enhanced_rules(user_data):
    """Enhanced rule-based system with better medical/science detection"""
    try:
        riasec = user_data.get('riasec', {})
        skills = user_data.get('skills_text', '').lower()
        courses = user_data.get('courses_text', '').lower()
        passion = user_data.get('passion_text', '').lower()
        work_style = user_data.get('work_style', '').lower()
        
        logger.info(f"🔍 Analyzing - Skills: {skills}, Passion: {passion}, Work: {work_style}")
        
        # Medical/Healthcare detection
        medical_keywords = ['medical', 'lab', 'hospital', 'health', 'patient', 'doctor', 'nurse', 'biology', 'chemistry', 'research']
        tech_keywords = ['programming', 'python', 'java', 'computer', 'software', 'code', 'ai', 'technology']
        business_keywords = ['business', 'management', 'marketing', 'sales', 'entrepreneur']
        creative_keywords = ['design', 'art', 'creative', 'drawing', 'painting']
        teaching_keywords = ['teaching', 'helping', 'people', 'community', 'care']
        
        # Calculate scores for different fields
        scores = {
            'Medical Science': 0,
            'Computer Science': 0, 
            'Business Administration': 0,
            'Architecture': 0,
            'Psychology': 0,
            'Education': 0
        }
        
        # Medical field scoring
        medical_score = sum(1 for word in medical_keywords if word in skills + courses + passion)
        if 'lab' in work_style or 'research' in work_style:
            medical_score += 2
        if riasec.get('I', 0):  # Investigative
            medical_score += 2
        scores['Medical Science'] = medical_score
        
        # Computer Science scoring
        tech_score = sum(1 for word in tech_keywords if word in skills + courses + passion)
        if riasec.get('I', 0):
            tech_score += 2
        scores['Computer Science'] = tech_score
        
        # Business scoring
        business_score = sum(1 for word in business_keywords if word in skills + courses + passion)
        if riasec.get('E', 0):  # Enterprising
            business_score += 2
        scores['Business Administration'] = business_score
        
        # Architecture/Arts scoring
        creative_score = sum(1 for word in creative_keywords if word in skills + courses + passion)
        if riasec.get('A', 0):  # Artistic
            creative_score += 2
        scores['Architecture'] = creative_score
        
        # Psychology/Education scoring
        teaching_score = sum(1 for word in teaching_keywords if word in skills + courses + passion)
        if riasec.get('S', 0):  # Social
            teaching_score += 2
        scores['Psychology'] = teaching_score
        scores['Education'] = teaching_score
        
        logger.info(f"📊 Field scores: {scores}")
        
        # Find best match
        best_major = max(scores, key=scores.get)
        best_score = scores[best_major]
        
        # Map to faculties
        faculty_map = {
            'Medical Science': 'Faculty of Medicine',
            'Computer Science': 'Faculty of Engineering',
            'Business Administration': 'Faculty of Business',
            'Architecture': 'Faculty of Fine Arts', 
            'Psychology': 'Faculty of Arts and Sciences',
            'Education': 'Faculty of Education'
        }
        
        major = best_major
        faculty = faculty_map.get(major, 'Faculty of Arts and Sciences')
        
        # Detect specific information
        detected_skills = []
        if any(word in skills for word in medical_keywords):
            detected_skills.extend(["Medical Knowledge", "Laboratory Skills"])
        if any(word in skills for word in tech_keywords):
            detected_skills.append("Technical Skills")
        if any(word in skills for word in business_keywords):
            detected_skills.append("Business Acumen")
            
        detected_courses = []
        if any(word in courses for word in ['chemistry', 'biology', 'science']):
            detected_courses.extend(["Chemistry", "Biology"])
        if any(word in courses for word in ['math', 'calculus']):
            detected_courses.append("Mathematics")
        if any(word in courses for word in ['computer', 'programming']):
            detected_courses.append("Computer Science")
            
        detected_passion = "Medical Field" if medical_score > 0 else "Technology" if tech_score > 0 else "Various Interests"
        
        confidence = "High" if best_score >= 3 else "Medium" if best_score >= 1 else "Low"
        
        result = {
            'major': major,
            'faculty': faculty,
            'degree': "Bachelor of Science" if major in ['Medical Science', 'Computer Science'] else "Bachelor of Arts",
            'campus': "Main Campus",
            'detected_info': {
                'detected_skills': detected_skills if detected_skills else ["Analytical Thinking"],
                'detected_courses': detected_courses if detected_courses else ["General Education"],
                'detected_passion': detected_passion
            },
            'success': True,
            'confidence': confidence,
            'method': 'Enhanced Rule-Based'
        }
        
        logger.info(f"✅ ENHANCED PREDICTION: {major} (score: {best_score})")
        return result
        
    except Exception as e:
        logger.error(f"❌ Enhanced prediction failed: {e}")
        # Fallback
        return {
            'major': "General Studies",
            'faculty': "Faculty of Arts and Sciences", 
            'degree': "Bachelor of Arts",
            'campus': "Main Campus",
            'detected_info': {
                'detected_skills': ["Critical Thinking"],
                'detected_courses': ["General Education"],
                'detected_passion': "Learning"
            },
            'success': True,
            'confidence': "Low",
            'method': 'Fallback'
        }

def predict_major(user_data):
    """
    Main prediction function - tries real ML first, then enhanced rules
    """
    logger.info("🤖 STARTING MAJOR PREDICTION")
    
    # Try REAL ML model first
    if ml_models_loaded:
        ml_result, ml_error = predict_with_real_ml(user_data)
        if ml_result:
            ml_result['success'] = True
            logger.info("🎯 USING REAL ML MODEL")
            return ml_result
        else:
            logger.warning(f"🔄 Real ML failed: {ml_error}")
    
    # Use enhanced rule-based system
    logger.info("🔄 Using enhanced rule-based system")
    result = predict_major_enhanced_rules(user_data)
    result['success'] = True
    return result

# Log final status
if ml_models_loaded:
    logger.info("🎉 SYSTEM READY - USING REAL ML MODELS!")
else:
    logger.info("🔄 SYSTEM READY - Using enhanced rule-based system")
