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
        
        if self.api_key:
            logger.info("✅ OpenAI API key loaded successfully")
            # Test the API key immediately
            self._test_openai_connection()
        else:
            logger.error("❌ OPENAI_API_KEY not found in environment")
        
        logger.info("✅ Career RAG class created")

    def _test_openai_connection(self):
        """Test OpenAI connection on startup"""
        try:
            test_prompt = "Hello, respond with 'OK' if you can hear me."
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": test_prompt}],
                "max_tokens": 5
            }
            
            response = requests.post(self.api_url, headers=headers, json=data, timeout=10)
            if response.status_code == 200:
                logger.info("✅ OpenAI API connection test: SUCCESS")
            else:
                logger.error(f"❌ OpenAI API test failed: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"❌ OpenAI API test error: {e}")

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

    def _find_relevant_qa(self, question, top_k=3):
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
            "model": "gpt-3.5-turbo",  # Using faster model
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 400
        }
        
        try:
            logger.info("🔄 Making OpenAI API request...")
            response = requests.post(self.api_url, headers=headers, json=data, timeout=20)
            logger.info(f"📡 Response status: {response.status_code}")
            
            if response.status_code == 401:
                logger.error("❌ OpenAI API: Invalid API Key")
                raise ValueError("Invalid OpenAI API Key")
            elif response.status_code == 429:
                logger.error("❌ OpenAI API: Rate Limit Exceeded")
                raise ValueError("OpenAI Rate Limit Exceeded")
            elif response.status_code != 200:
                logger.error(f"❌ OpenAI API Error {response.status_code}: {response.text}")
                response.raise_for_status()
            
            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip()
            logger.info("✅ OpenAI response received successfully")
            return answer
            
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

IMPORTANT: Provide a clear, professional answer based on the context above. Structure your response with proper paragraphs and focus on being helpful.

ANSWER:"""

            # Use direct OpenAI API call - FORCE it to work
            if not self.api_key:
                logger.error("❌ No API key available")
                raise ValueError("OpenAI API key not configured")
            
            logger.info("🚀 Calling OpenAI API...")
            answer = self._call_openai_direct(prompt)
            
            return {
                "answer": answer,
                "relevant_matches": len(relevant_data),
                "confidence": "High"
            }
            
        except Exception as e:
            logger.error(f"❌ RAG error: {e}")
            # Instead of falling back to dataset, return a clear error message
            return {
                "answer": "I'm currently experiencing technical difficulties. Please try again in a moment or contact support if the issue persists.",
                "confidence": "Low"
            }

    def _clean_text_for_prompt(self, text):
        """Clean text before sending to OpenAI"""
        if not text or pd.isna(text):
            return ""
        
        # Remove URLs and clean text
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'Continue Reading', '', text)
        
        # Take first 120 words
        words = text.split()[:120]
        return ' '.join(words)
