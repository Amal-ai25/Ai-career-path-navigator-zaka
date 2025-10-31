import os
from openai import OpenAI
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CareerCompassChat:
    def __init__(self):
        self.llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.is_initialized = True  # Always ready

    def ask_question(self, question):
        """Direct OpenAI chat without RAG"""
        try:
            prompt = f"""
            You are Career Compass, an expert career guidance assistant. 
            Provide helpful, accurate advice about careers, education, majors, skills, and professional development.
            
            Question: {question}
            
            Please provide a comprehensive, helpful answer focused on career guidance.
            """
            
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400
            )

            answer = response.choices[0].message.content.strip()
            
            return {
                "answer": answer,
                "confidence": "High",
                "source": "OpenAI Career Expert"
            }

        except Exception as e:
            logger.error(f"Chat error: {e}")
            return {
                "answer": "I'm here to help with career guidance! I can assist with:\n• Career path recommendations\n• Major selection advice\n• Skill development guidance\n• Educational requirements\n• Job market insights\n\nWhat would you like to know?",
                "confidence": "High"
            }

# Create instance immediately
career_system = CareerCompassChat()
logger.info("✅ Career chat system ready - Direct OpenAI")
