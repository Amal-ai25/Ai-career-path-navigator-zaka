import os
from openai import OpenAI
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CareerCompassChat:
    def __init__(self):
        try:
            # CORRECT OpenAI v1.3.0 initialization
            self.llm_client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
                # Remove any proxy parameters that might cause issues
            )
            self.is_initialized = True
            logger.info("✅ Career chat system initialized with GPT-4o-mini")
        except Exception as e:
            logger.error(f"OpenAI init error: {e}")
            self.is_initialized = False

    def ask_question(self, question):
        """Direct OpenAI chat with GPT-4o-mini"""
        try:
            if not self.is_initialized:
                return {
                    "answer": "🎓 Welcome to Career Compass! I'm currently optimizing my systems. Please try again in a moment.",
                    "confidence": "Medium"
                }

            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are Career Compass, an expert career guidance assistant. Provide helpful, accurate advice about careers, education, majors, skills, and professional development. Be comprehensive and practical."
                    },
                    {
                        "role": "user", 
                        "content": question
                    }
                ],
                temperature=0.3,
                max_tokens=400
            )

            answer = response.choices[0].message.content.strip()
            
            return {
                "answer": answer,
                "confidence": "High",
                "source": "GPT-4o-mini Career Expert"
            }

        except Exception as e:
            logger.error(f"Chat error: {e}")
            # Intelligent fallback based on question content
            question_lower = question.lower()
            
            if any(word in question_lower for word in ['career', 'job', 'work', 'profession']):
                return {
                    "answer": "🌟 For career guidance, I recommend:\n• Identify your strengths and interests\n• Research growing industries like tech, healthcare, and renewable energy\n• Consider both passion and job market demand\n• Network with professionals in fields you're interested in\n• Gain relevant experience through internships or projects",
                    "confidence": "High"
                }
            elif any(word in question_lower for word in ['major', 'study', 'degree', 'university']):
                return {
                    "answer": "🎓 Popular and high-demand majors include:\n• Computer Science & AI\n• Business Administration\n• Engineering (Various disciplines)\n• Healthcare & Medicine\n• Data Science\n• Environmental Science\n\nChoose based on your interests, skills, and career goals!",
                    "confidence": "High"
                }
            elif any(word in question_lower for word in ['skill', 'learn', 'develop']):
                return {
                    "answer": "💡 Essential skills for today's job market:\n• Technical: Programming, Data Analysis, Digital Literacy\n• Soft Skills: Communication, Problem-solving, Adaptability\n• Business: Project Management, Critical Thinking\n• Consider online courses, certifications, and practical projects to build these skills!",
                    "confidence": "High"
                }
            else:
                return {
                    "answer": "🎓 Welcome to Career Compass! I can help you with:\n• Career path recommendations\n• Major and degree selection\n• Skill development guidance\n• Educational planning\n• Job market insights\n\nWhat specific career or education questions can I help with today?",
                    "confidence": "High"
                }

# Create instance immediately
career_system = CareerCompassChat()
