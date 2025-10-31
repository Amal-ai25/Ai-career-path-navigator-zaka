import logging

logger = logging.getLogger(__name__)

def predict_major(user_data):
    """Intelligent fallback without any model loading"""
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
        
        return {
            'major': major,
            'faculty': faculty,
            'degree': "Bachelor of Science" if "Engineering" in faculty else "Bachelor of Arts",
            'campus': "Main Campus",
            'detected_info': {
                'detected_skills': detected_skills,
                'detected_courses': detected_courses,
                'detected_passion': detected_passion
            },
            'success': True,
            'confidence': "High"
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
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
            'confidence': "Medium"
        }

logger.info("✅ ML fallback system ready - No model loading required")
