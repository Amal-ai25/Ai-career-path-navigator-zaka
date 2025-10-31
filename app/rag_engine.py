import os
import pandas as pd
import openai
import logging
from dotenv import load_dotenv
import re

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CareerCompassRAG:
    def __init__(self):
        # Set API key directly
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.career_data = None
        self.is_initialized = False
        logger.info("✅ Career RAG class created")

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
        """Ask question with RAG using your dataset"""
        try:
            if not self.is_initialized or self.career_data is None:
                return {
                    "answer": "Career guidance system is starting up. Please try again in a moment.",
                    "confidence": "Low"
                }

            logger.info(f"🤔 Processing: {question}")

            # Find relevant Q&A from your dataset
            relevant_data = self._find_relevant_qa(question)
            
            if len(relevant_data) == 0:
                return {
                    "answer": "I don't have specific information about that topic. Please try asking about career paths, majors, skills, or education.",
                    "confidence": "Low"
                }

            # Build context from your dataset
            context = "RELEVANT CAREER INFORMATION:\n\n"
            for q, a in relevant_data:
                context += f"Q: {q}\nA: {a}\n\n"

            logger.info(f"🔍 Found {len(relevant_data)} relevant entries")

            # Create RAG prompt
            prompt = f"""
            You are Career Compass, a career guidance expert. 
            Use the career information below to answer the question.

            {context}

            QUESTION: {question}

            Provide accurate, specific career guidance based on the context.
            """

            # Use OpenAI
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400
            )

            answer = response.choices[0].message.content.strip()
            
            return {
                "answer": answer,
                "relevant_matches": len(relevant_data),
                "confidence": "High"
            }

        except Exception as e:
            logger.error(f"❌ RAG error: {e}")
            return self._get_fallback_from_dataset(question)

    def _get_fallback_from_dataset(self, question):
        """Fallback: return answers directly from dataset"""
        relevant_data = self._find_relevant_qa(question, top_k=3)
        
        if relevant_data:
            answer = "Based on career database:\n\n"
            for q, a in relevant_data:
                answer += f"• {a}\n"
            return {"answer": answer, "confidence": "Medium"}
        else:
            return {
                "answer": "I'm here to help with career guidance! Please ask about careers, education, majors, or skills.",
                "confidence": "Medium"
            }
