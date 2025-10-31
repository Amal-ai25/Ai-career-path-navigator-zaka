import os
import pandas as pd
from openai import OpenAI
import logging
from dotenv import load_dotenv
import re

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CareerCompassChat:
    def __init__(self):
        self.llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.career_data = None
        self.is_initialized = False
        logger.info("✅ Career chat system initializing")

    def _load_career_data(self, data_path):
        """Load career data"""
        try:
            if not os.path.exists(data_path):
                logger.error(f"❌ Dataset not found: {data_path}")
                return False
            
            self.career_data = pd.read_csv(data_path)
            logger.info(f"✅ Loaded {len(self.career_data)} career Q&A pairs")
            return True
        except Exception as e:
            logger.error(f"❌ Data loading error: {e}")
            return False

    def initialize_system(self, data_path):
        """Initialize the system"""
        logger.info("🚀 Initializing Career System...")
        
        if self._load_career_data(data_path):
            self.is_initialized = True
            logger.info("🎉 Career system initialized successfully!")
            return True
        return False

    def _find_relevant_content(self, question):
        """Simple keyword matching for relevant content"""
        if self.career_data is None:
            return []
        
        question_lower = question.lower()
        question_words = set(re.findall(r'\b\w+\b', question_lower))
        
        matches = []
        
        for idx, row in self.career_data.iterrows():
            if pd.notna(row['question']) and pd.notna(row['answer']):
                q_text = str(row['question']).lower()
                a_text = str(row['answer']).lower()
                
                # Simple keyword matching
                q_match = sum(1 for word in question_words if word in q_text)
                a_match = sum(1 for word in question_words if word in a_text)
                
                total_score = q_match * 3 + a_match
                
                if total_score > 0:
                    matches.append((total_score, row['question'], row['answer']))
        
        # Sort by score and return top 3
        matches.sort(reverse=True, key=lambda x: x[0])
        return [(q, a) for _, q, a in matches[:3]]

    def ask_question(self, question):
        """Ask question with career context"""
        try:
            if not self.is_initialized:
                return self._get_smart_response(question)

            logger.info(f"🤔 Processing: {question}")

            # Find relevant content
            relevant_content = self._find_relevant_content(question)
            
            if relevant_content:
                # Build context from relevant content
                context = "Related career information:\n"
                for q, a in relevant_content:
                    context += f"Q: {q}\nA: {a}\n\n"
                
                prompt = f"""
                You are Career Compass, a career guidance expert.
                
                {context}
                
                Based on the above information and your expertise, answer this question:
                {question}
                
                Provide helpful, specific career guidance.
                """
            else:
                # No relevant content found, use career expert mode
                prompt = f"""
                You are Career Compass, a career guidance expert specializing in:
                - Career paths and job markets
                - University majors and education  
                - Skill development
                - Professional advice
                
                Question: {question}
                
                Provide comprehensive career guidance.
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
                "confidence": "High"
            }

        except Exception as e:
            logger.error(f"Chat error: {e}")
            return self._get_smart_response(question)

    def _get_smart_response(self, question):
        """Smart fallback responses"""
        question_lower = question.lower()
        
        # Smart keyword responses
        if any(word in question_lower for word in ['law', 'legal', 'attorney']):
            return {
                "answer": "📚 **Law Specializations**:\n• Criminal Law (prosecution/defense)\n• Corporate Law (business legal matters)  \n• Family Law (divorce, custody)\n• Intellectual Property (patents, copyrights)\n• International Law (cross-border issues)\n• Environmental Law (regulations, sustainability)\n\nEach specialization offers different career paths and requires specific skills.",
                "confidence": "High"
            }
        
        elif any(word in question_lower for word in ['computer', 'programming', 'software', 'ai']):
            return {
                "answer": "💻 **Computer Science Careers**:\n• Software Development (web, mobile, desktop)\n• Data Science & AI (machine learning, analytics)\n• Cybersecurity (network protection, ethical hacking)\n• Cloud Computing (AWS, Azure, Google Cloud)\n• Game Development (design, programming)\n\nKey skills: Programming, algorithms, problem-solving, teamwork.",
                "confidence": "High"
            }
        
        elif any(word in question_lower for word in ['business', 'management', 'marketing', 'finance']):
            return {
                "answer": "💼 **Business Specializations**:\n• Management (leadership, operations)\n• Marketing (digital, brand management)\n• Finance (banking, investment, analysis)\n• Entrepreneurship (startups, innovation)\n• Human Resources (talent management)\n\nThese lead to diverse careers across all industries.",
                "confidence": "High"
            }
        
        elif any(word in question_lower for word in ['engineering', 'civil', 'mechanical', 'electrical']):
            return {
                "answer": "⚙️ **Engineering Fields**:\n• Civil Engineering (infrastructure, construction)\n• Mechanical Engineering (machines, systems)\n• Electrical Engineering (electronics, power)\n• Computer Engineering (hardware, software)\n• Chemical Engineering (processes, materials)\n\nEach requires strong math and problem-solving skills.",
                "confidence": "High"
            }
        
        elif any(word in question_lower for word in ['skill', 'learn', 'develop']):
            return {
                "answer": "🎯 **Essential Career Skills**:\n\n**Technical Skills**:\n• Programming (Python, Java, JavaScript)\n• Data Analysis (Excel, SQL, Statistics)\n• Digital Marketing (SEO, Social Media)\n• Project Management\n\n**Soft Skills**:\n• Communication & Presentation\n• Problem-solving & Critical Thinking\n• Teamwork & Collaboration\n• Adaptability & Learning\n\nContinuous skill development is key for career growth!",
                "confidence": "High"
            }
        
        elif any(word in question_lower for word in ['major', 'study', 'degree', 'university']):
            return {
                "answer": "🎓 **Popular University Majors**:\n\n**High Demand Fields**:\n• Computer Science & AI\n• Business Administration\n• Engineering (Various types)\n• Healthcare & Medicine\n• Data Science\n• Environmental Science\n\n**Choosing Tips**:\n• Consider your interests and strengths\n• Research job market demand\n• Look at career growth opportunities\n• Consider required education duration\n• Talk to professionals in the field",
                "confidence": "High"
            }
        
        else:
            return {
                "answer": "🎓 **Welcome to Career Compass!** 🤖\n\nI can help you with:\n• Career path recommendations\n• University major selection\n• Skill development guidance\n• Educational requirements\n• Job market insights\n• Professional development\n\n**Try asking about**:\n• 'What are the best careers in tech?'\n• 'Which engineering field should I choose?'\n• 'What skills are important for business?'\n• 'Tell me about law specializations'\n\nWhat career questions can I help with today?",
                "confidence": "High"
            }

# Create instance
career_system = CareerCompassChat()
