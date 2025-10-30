import logging

logger = logging.getLogger(__name__)

def predict_major(user_data):
    """Temporary fallback while ML models are fixed"""
    return {
        "major": "Computer Science",
        "faculty": "Faculty of Engineering",
        "degree": "Bachelor of Science", 
        "campus": "Main Campus",
        "detected_info": {
            "detected_skills": ["temporary"],
            "detected_courses": ["temporary"],
            "detected_passion": "technology"
        },
        "success": True,
        "note": "ML system upgrading - showing sample recommendation"
    }

logger.info("🔄 ML system in maintenance mode - using fallback")
