import os
import pandas as pd
import logging
from dotenv import load_dotenv
import re
import requests
import json

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CareerCompassRAG:
    def __init__(self):
        # Use direct API calls instead of OpenAI client
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.career_data = None
        self.is_initialized = False
        
        if not self.api_key:
            logger.warning("⚠️ OPENAI_API_KEY not found")
        else:
            logger.info("✅ OpenAI API configured for direct requests")
        
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

    def _call_openai_direct(self, prompt):
        """Make direct HTTP request to OpenAI API"""
        if not self.api_key:
            raise ValueError("OpenAI API key not available")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 500
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ OpenAI API request failed: {e}")
            raise
        except KeyError as e:
            logger.error(f"❌ Unexpected response format from OpenAI: {e}")
            raise

    def ask_question(self, question):
        """Ask question with proper RAG using OpenAI"""
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
            context = "RELEVANT CAREER GUIDANCE INFORMATION:\n\n"
            for i, (q, a) in enumerate(relevant_data, 1):
                # Clean the answers before sending to OpenAI
                clean_a = self._clean_text_for_prompt(a)
                context += f"{i}. Q: {q}\n   A: {clean_a}\n\n"

            logger.info(f"🔍 Found {len(relevant_data)} relevant entries")

            # Create optimized RAG prompt
            prompt = f"""You are Career Compass, an expert career guidance advisor. Use the following career information from our database to provide a helpful, professional answer.

{context}

USER QUESTION: {question}

IMPORTANT INSTRUCTIONS:
- Provide specific, actionable career guidance based on the context above
- Focus on being helpful and professional
- Keep your answer concise (2-3 paragraphs maximum)
- If the context doesn't fully address the question, provide general career advice
- Use a warm, supportive tone
- Structure your answer with clear paragraphs

ANSWER:"""

            # Use direct OpenAI API call
            if self.api_key:
                answer = self._call_openai_direct(prompt)
                logger.info("✅ Successfully generated answer with OpenAI")
            else:
                raise ValueError("OpenAI API key not available")
            
            return {
                "answer": answer,
                "relevant_matches": len(relevant_data),
                "confidence": "High"
            }

        except Exception as e:
            logger.error(f"❌ RAG error: {e}")
            # Enhanced fallback
            return self._get_enhanced_fallback(question)

    def _clean_text_for_prompt(self, text):
        """Clean text before sending to OpenAI to improve response quality"""
        if not text or pd.isna(text):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special formatting markers
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)
        
        # Take first 200 words to avoid overly long context
        words = text.split()[:200]
        return ' '.join(words)

    def _get_enhanced_fallback(self, question):
        """Enhanced fallback when OpenAI fails"""
        relevant_data = self._find_relevant_qa(question, top_k=2)
        
        if relevant_data:
            # Use the best matching answer with cleaning
            best_q, best_a = relevant_data[0]
            clean_answer = self._clean_text_for_prompt(best_a)
            
            if len(clean_answer) > 50:
                return {
                    "answer": f"🎯 Based on career guidance information:\n\n{clean_answer}\n\n*Note: Using database information directly*",
                    "confidence": "Medium"
                }
        
        # Default professional response
        return {
            "answer": "👋 I'm Career Compass! I specialize in helping students and professionals with career guidance, education paths, and skill development.\n\nPlease ask me about:\n• Career options and paths\n• College majors and degrees  \n• Skills development\n• Work preferences\n\nWhat career question can I help you with today?",
            "confidence": "Medium"
        }
