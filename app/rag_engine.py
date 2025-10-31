import os
import pandas as pd
import logging
from dotenv import load_dotenv
import re
import requests
import json
import time

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CareerCompassRAG:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.career_data = None
        self.is_initialized = False
        
        logger.info(f"🔑 API Key available: {bool(self.api_key)}")
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

    def _find_relevant_qa(self, question, top_k=3):  # Reduced from 5 to 3
        """Find relevant Q&A using keyword matching"""
        if self.career_data is None:
            return []
        
        start_time = time.time()
        question_lower = question.lower()
        question_words = set(re.findall(r'\b\w+\b', question_lower))
        
        matches = []
        
        for idx, row in self.career_data.iterrows():
            # Add timeout check during data processing
            if time.time() - start_time > 5.0:  # 5 second timeout for data processing
                logger.warning("⚠️ Data processing timeout, returning partial results")
                break
                
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
        """Make direct HTTP request to OpenAI API with timeout"""
        if not self.api_key:
            logger.error("❌ No API key available")
            raise ValueError("OpenAI API key not available")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": "gpt-3.5-turbo",  # Use faster model
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 300  # Reduced from 500
        }
        
        try:
            logger.info("🔄 Making OpenAI API request...")
            response = requests.post(self.api_url, headers=headers, json=data, timeout=15.0)  # 15s timeout
            logger.info(f"📡 Response status: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"❌ API Error {response.status_code}: {response.text}")
                response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
            
        except requests.exceptions.Timeout:
            logger.error("❌ OpenAI API timeout")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ OpenAI API request failed: {e}")
            raise
        except KeyError as e:
            logger.error(f"❌ Unexpected response format from OpenAI: {e}")
            raise

    def ask_question(self, question):
        """Ask question with proper RAG using OpenAI"""
        start_time = time.time()
        
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
                clean_a = self._clean_text_for_prompt(a)
                context += f"{i}. Q: {q}\n   A: {clean_a}\n\n"

            logger.info(f"🔍 Found {len(relevant_data)} relevant entries")

            # Create optimized RAG prompt
            prompt = f"""You are Career Compass, an expert career guidance advisor. Use the specific career information from our database below to provide a detailed, actionable answer.

CONTEXT FROM CAREER DATABASE:
{context}

USER QUESTION: {question}

Provide specific, actionable career guidance based on the context above. Keep it concise (2 paragraphs maximum).

ANSWER:"""

            # Use direct OpenAI API call with fallback
            if self.api_key:
                logger.info("🚀 Attempting to call OpenAI API...")
                answer = self._call_openai_direct(prompt)
                total_time = time.time() - start_time
                logger.info(f"✅ Successfully generated answer in {total_time:.2f}s")
                return {
                    "answer": answer,
                    "relevant_matches": len(relevant_data),
                    "confidence": "High"
                }
            else:
                logger.error("❌ No API key available for OpenAI")
                raise ValueError("OpenAI API key not available")
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"❌ RAG error after {total_time:.2f}s: {e}")
            logger.info("🔄 Falling back to dataset mode")
            # Quick fallback
            return self._get_quick_fallback(question, relevant_data)

    def _clean_text_for_prompt(self, text):
        """Clean text before sending to OpenAI"""
        if not text or pd.isna(text):
            return ""
        
        # Quick cleaning for performance
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Take first 100 words for performance
        words = text.split()[:100]
        return ' '.join(words)

    def _get_quick_fallback(self, question, relevant_data=None):
        """Quick fallback when OpenAI fails"""
        if relevant_data is None:
            relevant_data = self._find_relevant_qa(question, top_k=1)  # Only 1 for speed
        
        if relevant_data:
            best_q, best_a = relevant_data[0]
            clean_answer = self._clean_text_for_prompt(best_a)
            
            if len(clean_answer) > 30:
                return {
                    "answer": f"Based on career guidance:\n\n{clean_answer}",
                    "confidence": "Medium"
                }
        
        # Quick default response
        return {
            "answer": "I'm here to help with career guidance! Ask me about careers, education, or skills.",
            "confidence": "Medium"
        }
