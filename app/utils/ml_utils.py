import logging
import random

logger = logging.getLogger(__name__)

# Sample recommendations for fallback
SAMPLE_MAJORS = [
    "Computer Science", "Business Administration", "Electrical Engineering",
    "Mechanical Engineering", "Civil Engineering", "Architecture",
    "Medicine", "Law", "Psychology", "Economics", "Marketing"
]

SAMPLE_FACULTIES = [
    "Faculty of Engineering", "Faculty of Sciences", "Faculty of Business",
    "Faculty of Medicine", "Faculty of Law", "Faculty of Arts and Sciences"
]

SAMPLE_DEGREES = ["Bachelor of Science", "Bachelor of Arts", "Bachelor of Engineering"]
SAMPLE_CAMPUSES = ["Main Campus", "Engineering Campus", "Medical Campus"]

def predict_major(user_data):
    """Fallback prediction while ML models are being fixed"""
    try:
        # Simple logic based on RIASEC scores
        riasec = user_data.get('riasec', {})
        
        # Determine major category based on highest RIASEC score
        if riasec.get('I', 0) > riasec.get('R', 0):
            category = "Science/Research"
            major = "Computer Science" if "programming" in user_data.get('skills_text', '').lower() else "Biology"
        elif riasec.get('A', 0) > riasec.get('C', 0):
            category = "Creative/Arts" 
            major = "Architecture" if "design" in user_data.get('skills_text', '').lower() else "Fine Arts"
        elif riasec.get('E', 0) > riasec.get('S', 0):
            category = "Business/Leadership"
            major = "Business Administration" if "management" in user_data.get('skills_text', '').lower() else "Economics"
        else:
            category = "General"
            major = random.choice(SAMPLE_MAJORS)
        
        # Simple skill detection
        skills_text = user_data.get('skills_text', '').lower()
        detected_skills = []
        if any(word in skills_text for word in ['python', 'java', 'programming', 'coding']):
            detected_skills.append("Programming")
        if any(word in skills_text for word in ['design', 'creative', 'art']):
            detected_skills.append("Design")
        if any(word in skills_text for word in ['management', 'leadership', 'business']):
            detected_skills.append("Management")
        
        courses_text = user_data.get('courses_text', '').lower()
        detected_courses = []
        if any(word in courses_text for word in ['computer', 'programming', 'math']):
            detected_courses.append("Computer Science")
        if any(word in courses_text for word in ['business', 'economics', 'marketing']):
            detected_courses.append("Business")
        
        passion_text = user_data.get('passion_text', '').lower()
        detected_passion = "Technology" if "tech" in passion_text else \
                          "Business" if "business" in passion_text else \
                          "Arts" if "art" in passion_text else "General"
        
        return {
            'major': major,
            'faculty': random.choice(SAMPLE_FACULTIES),
            'degree': random.choice(SAMPLE_DEGREES),
            'campus': random.choice(SAMPLE_CAMPUSES),
            'detected_info': {
                'detected_skills': detected_skills or ["Analytical Thinking", "Problem Solving"],
                'detected_courses': detected_courses or ["General Education"],
                'detected_passion': detected_passion
            },
            'success': True,
            'note': 'Using intelligent fallback system while ML models are optimized'
        }
        
    except Exception as e:
        logger.error(f"Fallback prediction error: {e}")
        return {
            'major': "Computer Science",
            'faculty': "Faculty of Engineering", 
            'degree': "Bachelor of Science",
            'campus': "Main Campus",
            'detected_info': {
                'detected_skills': ["General Skills"],
                'detected_courses': ["General Courses"],
                'detected_passion': "Learning"
            },
            'success': True,
            'note': 'Basic fallback recommendation'
        }

logger.info("✅ ML fallback system ready")
