import os
import pandas as pd
import logging
from dotenv import load_dotenv
import re
import random

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CareerCompassRAG:
    def __init__(self):
        # Simple initialization without OpenAI for Render compatibility
        self.career_data = None
        self.is_initialized = False
        logger.info("✅ Career RAG class created (Enhanced Dataset Mode)")

    def _load_career_data(self, data_path):
        """Load career dataset"""
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
        """Initialize RAG system"""
        logger.info("🚀 Initializing Career RAG...")
        
        if self._load_career_data(data_path):
            self.is_initialized = True
            logger.info("🎉 Career RAG initialized successfully!")
            return True
        return False

    def _find_relevant_qa(self, question, top_k=5):
        """Find relevant Q&A using keyword matching"""
        if self.career_data is None:
            return []
        
        question_lower = question.lower()
        question_words = set(re.findall(r'\b\w+\b', question_lower))
        
        matches = []
        
        for idx, row in self.career_data.iterrows():
            if pd.notna(row['question']) and pd.notna(row['answer']):
                q_text = str(row['question']).lower()
                a_text = str(row['answer']).lower()
                
                # Calculate match score
                q_match = sum(1 for word in question_words if word in q_text)
                a_match = sum(1 for word in question_words if word in a_text)
                
                total_score = q_match * 3 + a_match
                
                if total_score > 0:
                    matches.append((total_score, row['question'], row['answer']))
        
        # Sort by score and return top matches
        matches.sort(reverse=True, key=lambda x: x[0])
        return [(q, a) for _, q, a in matches[:top_k]]

    def ask_question(self, question):
        """Ask question with intelligent response generation"""
        try:
            if not self.is_initialized or self.career_data is None:
                return {
                    "answer": "🎓 Career guidance system is starting up. Please try again in a moment.",
                    "confidence": "Low"
                }

            logger.info(f"🤔 Processing: {question}")

            # Find relevant Q&A from your dataset
            relevant_data = self._find_relevant_qa(question, top_k=3)
            
            if len(relevant_data) == 0:
                return self._get_default_response(question)

            # Create intelligent response from dataset
            response = self._create_intelligent_response(question, relevant_data)
            return response

        except Exception as e:
            logger.error(f"❌ RAG error: {e}")
            return self._get_default_response(question)

    def _create_intelligent_response(self, question, relevant_data):
        """Create intelligent, well-formatted responses from dataset matches"""
        
        # Clean and process the best matching answers
        cleaned_answers = []
        for q, a in relevant_data[:2]:  # Use top 2 most relevant
            # Clean the answer - remove extra spaces, special formatting
            clean_a = re.sub(r'\s+', ' ', a).strip()
            clean_a = clean_a[:400]  # Limit length
            
            # Skip if answer is too short or low quality
            if len(clean_a) > 50 and not clean_a.startswith('http'):
                cleaned_answers.append(clean_a)
        
        if not cleaned_answers:
            return self._get_default_response(question)
        
        # Create a professional response
        if len(cleaned_answers) == 1:
            answer = f"🎯 **Career Guidance:**\n\n{cleaned_answers[0]}"
        else:
            # Combine multiple relevant answers intelligently
            main_answer = cleaned_answers[0]
            additional_insight = cleaned_answers[1] if len(cleaned_answers) > 1 else ""
            
            answer = f"🎯 **Career Guidance:**\n\n{main_answer}"
            if additional_insight and len(additional_insight) > 30:
                answer += f"\n\n💡 **Additional Insight:**\n{additional_insight}"
        
        # Add professional footer
        answer += "\n\n---\n*Based on comprehensive career guidance database*"
        
        return {
            "answer": answer,
            "confidence": "High",
            "relevant_matches": len(relevant_data)
        }

    def _get_default_response(self, question):
        """Get default responses for common questions"""
        question_lower = question.lower()
        
        # Smart default responses based on question type
        if any(word in question_lower for word in ['hello', 'hi', 'hey', 'hola']):
            return {
                "answer": "👋 Hello! I'm Career Compass, your career guidance assistant. I can help you with:\n\n• Career paths and options\n• College majors and education\n• Skills development\n• Work style preferences\n\nWhat career question can I help you with today?",
                "confidence": "High"
            }
        elif any(word in question_lower for word in ['career', 'job', 'profession']):
            return {
                "answer": "🎓 **Career Guidance Available:**\n\nI can help you explore various career paths based on your interests, skills, and preferences. Tell me about:\n\n• Your favorite subjects or activities\n• Skills you enjoy using\n• Work environment preferences\n• Long-term goals\n\nWhat areas are you most interested in?",
                "confidence": "High"
            }
        elif any(word in question_lower for word in ['major', 'degree', 'college', 'university']):
            return {
                "answer": "📚 **Education Path Guidance:**\n\nI can help you choose the right major based on:\n\n• Your interests (RIASEC personality type)\n• Favorite subjects and skills\n• Preferred work style\n• Career goals\n\nTry our major prediction tool or tell me what subjects you enjoy!",
                "confidence": "High"
            }
        elif any(word in question_lower for word in ['skill', 'learn', 'develop']):
            return {
                "answer": "🛠️ **Skills Development:**\n\nI can guide you on developing skills for various careers. Consider:\n\n• Technical skills (programming, design, analysis)\n• Soft skills (communication, leadership)\n• Industry-specific competencies\n\nWhat career field are you interested in?",
                "confidence": "High"
            }
        else:
            return {
                "answer": "🎓 **Career Compass Assistant**\n\nI specialize in career guidance and education planning. You can ask me about:\n\n• Career options and paths\n• College majors and degrees\n• Skills development\n• Work preferences\n• Education requirements\n\nWhat specific career or education question can I help you with?",
                "confidence": "Medium"
            }

    def _clean_answer_text(self, text):
        """Clean and format answer text"""
        if not text or pd.isna(text):
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', str(text))
        
        # Remove URLs and special formatting
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'\[.*?\]', '', text)
        
        # Ensure proper sentence structure
        sentences = text.split('.')
        clean_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Only keep meaningful sentences
                # Capitalize first letter
                if sentence and not sentence[0].isupper():
                    sentence = sentence[0].upper() + sentence[1:]
                clean_sentences.append(sentence)
        
        # Join back with proper punctuation
        cleaned_text = '. '.join(clean_sentences[:3])  # Max 3 sentences
        if cleaned_text and not cleaned_text.endswith('.'):
            cleaned_text += '.'
            
        return cleaned_text
