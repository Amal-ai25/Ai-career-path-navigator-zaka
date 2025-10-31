import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CareerCompassChat:
    def __init__(self):
        self.is_initialized = True
        logger.info("✅ Career chat system ready")

    def ask_question(self, question):
        """Smart career responses without OpenAI"""
        try:
            question_lower = question.lower()
            
            # Comprehensive career responses
            if any(word in question_lower for word in ['law', 'legal', 'attorney', 'lawyer']):
                return {
                    "answer": "⚖️ **Law Degree Specializations**:\n\n**Criminal Law**: Prosecution and defense of criminal cases\n**Corporate Law**: Business legal matters, contracts, compliance\n**Family Law**: Divorce, child custody, adoption\n**Intellectual Property**: Patents, trademarks, copyrights\n**International Law**: Cross-border legal issues, treaties\n**Environmental Law**: Environmental regulations, sustainability\n**Tax Law**: Taxation matters for individuals and businesses\n**Real Estate Law**: Property transactions, zoning\n\n**Career Paths**: Lawyer, Judge, Legal Consultant, Corporate Counsel",
                    "confidence": "High"
                }
            
            elif any(word in question_lower for word in ['computer', 'programming', 'software', 'ai', 'coding']):
                return {
                    "answer": "💻 **Computer Science Careers**:\n\n**Software Development**:\n• Web Development (Frontend/Backend)\n• Mobile App Development\n• Desktop Applications\n\n**AI & Data Science**:\n• Machine Learning Engineering\n• Data Analysis\n• Artificial Intelligence\n\n**Other Fields**:\n• Cybersecurity\n• Cloud Computing\n• Game Development\n• DevOps\n\n**Key Skills**: Python, Java, JavaScript, Algorithms, Problem-solving, Teamwork\n**Job Market**: High demand with excellent growth potential",
                    "confidence": "High"
                }
            
            elif any(word in question_lower for word in ['business', 'management', 'marketing', 'finance', 'mba']):
                return {
                    "answer": "💼 **Business Administration Specializations**:\n\n**Management**: Leadership, operations, strategy\n**Marketing**: Digital marketing, brand management, market research\n**Finance**: Banking, investment, financial analysis\n**Entrepreneurship**: Startups, innovation, business development\n**Human Resources**: Talent management, recruitment, training\n**Supply Chain**: Logistics, operations, procurement\n\n**Career Opportunities**: Manager, Consultant, Analyst, Entrepreneur, Executive",
                    "confidence": "High"
                }
            
            elif any(word in question_lower for word in ['engineering', 'civil', 'mechanical', 'electrical', 'chemical']):
                return {
                    "answer": "⚙️ **Engineering Fields**:\n\n**Civil Engineering**: Infrastructure, construction, buildings\n**Mechanical Engineering**: Machines, manufacturing, systems\n**Electrical Engineering**: Electronics, power systems, circuits\n**Computer Engineering**: Hardware, software integration\n**Chemical Engineering**: Processes, materials, manufacturing\n**Biomedical Engineering**: Medical devices, healthcare technology\n\n**Required Skills**: Mathematics, Physics, Problem-solving, Technical drawing\n**Job Prospects**: Strong demand across all engineering disciplines",
                    "confidence": "High"
                }
            
            elif any(word in question_lower for word in ['medicine', 'doctor', 'medical', 'health', 'nursing']):
                return {
                    "answer": "🏥 **Medical & Healthcare Careers**:\n\n**Doctors**:\n• General Practitioner\n• Surgeon\n• Pediatrician\n• Cardiologist\n\n**Other Medical Professionals**:\n• Dentist\n• Pharmacist\n• Nurse\n• Physical Therapist\n\n**Healthcare Administration**:\n• Hospital Management\n• Healthcare Consulting\n• Medical Research\n\n**Education**: Extensive training required (6+ years)\n**Specializations**: Numerous fields available after basic medical education",
                    "confidence": "High"
                }
            
            elif any(word in question_lower for word in ['skill', 'learn', 'develop', 'ability']):
                return {
                    "answer": "🎯 **Essential Career Skills for 2024**:\n\n**Technical Skills**:\n• Programming (Python, JavaScript, Java)\n• Data Analysis & Statistics\n• Digital Marketing\n• Project Management\n• Cloud Computing (AWS, Azure)\n\n**Soft Skills**:\n• Communication & Presentation\n• Critical Thinking & Problem-solving\n• Teamwork & Collaboration\n• Adaptability & Learning Agility\n• Leadership & Management\n\n**Industry-Specific Skills**:\n• AI & Machine Learning (Tech)\n• Financial Analysis (Business)\n• Clinical Skills (Healthcare)\n• Design Thinking (Creative Fields)\n\n**Tip**: Continuous learning is essential for career growth!",
                    "confidence": "High"
                }
            
            elif any(word in question_lower for word in ['major', 'study', 'degree', 'university', 'college']):
                return {
                    "answer": "🎓 **Choosing a University Major**:\n\n**High-Demand Fields**:\n• Computer Science & AI\n• Business Administration\n• Engineering (All types)\n• Healthcare & Medicine\n• Data Science\n• Environmental Science\n\n**Consider These Factors**:\n1. Your interests and passions\n2. Job market demand and growth\n3. Salary potential and career paths\n4. Required education duration\n5. Your academic strengths\n\n**Popular Majors**:\n• Computer Science\n• Business Administration\n• Psychology\n• Engineering\n• Nursing\n• Biology\n• Economics\n\n**Advice**: Research each field thoroughly and talk to professionals!",
                    "confidence": "High"
                }
            
            elif any(word in question_lower for word in ['career', 'job', 'work', 'profession', 'occupation']):
                return {
                    "answer": "🌟 **Career Guidance**:\n\n**Steps to Choose a Career**:\n1. Self-assessment: Identify your interests and strengths\n2. Research: Explore different industries and roles\n3. Education: Understand required qualifications\n4. Experience: Gain practical experience through internships\n5. Network: Connect with professionals in your field of interest\n\n**Growing Industries**:\n• Technology & AI\n• Healthcare & Biotechnology\n• Renewable Energy\n• Data Science & Analytics\n• Digital Marketing\n• Cybersecurity\n\n**Career Development Tips**:\n• Continuously update your skills\n• Build a professional network\n• Seek mentorship\n• Consider work-life balance\n• Plan for long-term growth",
                    "confidence": "High"
                }
            
            elif any(word in question_lower for word in ['salary', 'pay', 'income', 'earn']):
                return {
                    "answer": "💰 **Career Earnings Overview**:\n\n**High-Earning Fields**:\n• Technology (Software Engineers, Data Scientists)\n• Healthcare (Doctors, Surgeons)\n• Engineering (Various disciplines)\n• Business (Executives, Consultants)\n• Law (Corporate Lawyers)\n\n**Factors Affecting Salary**:\n• Education level and specialization\n• Years of experience\n• Geographic location\n• Industry and company size\n• Specific skills and certifications\n\n**Note**: Salary research should consider cost of living and career satisfaction beyond just earnings.",
                    "confidence": "High"
                }
            
            else:
                return {
                    "answer": "🎓 **Welcome to Career Compass!** 🤖\n\nI'm your career guidance assistant! Here's what I can help you with:\n\n**Career Fields**:\n• Law & Legal professions\n• Computer Science & Technology\n• Business & Management\n• Engineering (All types)\n• Healthcare & Medicine\n• And many more!\n\n**Ask me about**:\n• 'What are the specializations in law?'\n• 'Which engineering field is best for me?'\n• 'What skills do I need for tech careers?'\n• 'How to choose a university major?'\n• 'Career options in business administration'\n\nWhat career questions can I help you explore today?",
                    "confidence": "High"
                }
                
        except Exception as e:
            logger.error(f"Response error: {e}")
            return {
                "answer": "🎓 Career Compass is here to help you navigate career choices and educational paths! I provide guidance on majors, skills, and career development. What specific area would you like to explore?",
                "confidence": "High"
            }

# Create instance - NO INITIALIZATION NEEDED
career_system = CareerCompassChat()
