import os
import pandas as pd
import numpy as np
from openai import OpenAI
import logging
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import re

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CareerCompassRAG:
    def __init__(self):
        self.llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = None
        self.career_data = None
        self.is_initialized = False
        logger.info("✅ Career RAG system initializing")

    def _load_embedding_model(self):
        """Load lightweight embedding model"""
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Embedding model loaded")
        except Exception as e:
            logger.error(f"❌ Embedding model error: {e}")
            self.embedding_model = None

    def _load_career_data(self, data_path):
        """Load and prepare career data"""
        try:
            if not os.path.exists(data_path):
                logger.error(f"❌ Dataset not found: {data_path}")
                return False
            
            self.career_data = pd.read_csv(data_path)
            logger.info(f"✅ Loaded {len(self.career_data)} career Q&A pairs")
            
            # Create embeddings for all answers
            if self.embedding_model:
                self.career_data['embedding'] = self.career_data['answer'].apply(
                    lambda x: self.embedding_model.encode(str(x))
                )
                logger.info("✅ Career data embeddings created")
            
            return True
        except Exception as e:
            logger.error(f"❌ Data loading error: {e}")
            return False

    def initialize_system(self, data_path):
        """Initialize the RAG system"""
        logger.info("🚀 Initializing Career RAG...")
        
        self._load_embedding_model()
        if self._load_career_data(data_path):
            self.is_initialized = True
            logger.info("🎉 Career RAG initialized successfully!")
            return True
        return False

    def _find_similar_questions(self, question, top_k=5):
        """Find similar questions using semantic search"""
        if not self.embedding_model or self.career_data is None:
            return []
        
        try:
            # Encode the question
            question_embedding = self.embedding_model.encode(question)
            
            # Calculate similarities
            similarities = []
            for idx, row in self.career_data.iterrows():
                if pd.notna(row['embedding']):
                    similarity = np.dot(question_embedding, row['embedding'])
                    similarities.append((similarity, idx))
            
            # Get top matches
            similarities.sort(reverse=True)
            top_indices = [idx for _, idx in similarities[:top_k]]
            
            return self.career_data.iloc[top_indices]
        except Exception as e:
            logger.error(f"Similarity search error: {e}")
            return []

    def _find_keyword_matches(self, question, top_k=5):
        """Fallback: keyword-based matching"""
        if self.career_data is None:
            return []
        
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        matches = []
        
        for idx, row in self.career_data.iterrows():
            if pd.notna(row['question']):
                question_text = str(row['question']).lower()
                answer_text = str(row['answer']).lower()
                
                # Calculate match score
                question_match = len(question_words.intersection(set(re.findall(r'\b\w+\b', question_text))))
                content_match = len(question_words.intersection(set(re.findall(r'\b\w+\b', answer_text))))
                
                total_score = question_match * 2 + content_match  # Weight question matches higher
                
                if total_score > 0:
                    matches.append((total_score, idx))
        
        matches.sort(reverse=True)
        return self.career_data.iloc[[idx for _, idx in matches[:top_k]]]

    def ask_question(self, question):
        """Ask question with RAG from your dataset"""
        try:
            if not self.is_initialized or self.career_data is None:
                return {
                    "answer": "Career guidance system is starting up. Please try again in a moment.",
                    "confidence": "Low"
                }

            logger.info(f"🤔 Processing: {question}")

            # Try semantic search first, then keyword fallback
            similar_data = self._find_similar_questions(question)
            if len(similar_data) == 0:
                similar_data = self._find_keyword_matches(question)
            
            if len(similar_data) == 0:
                # No relevant data found, use direct OpenAI
                return self._ask_direct_openai(question)

            # Build context from similar questions
            context = ""
            for _, row in similar_data.iterrows():
                context += f"Q: {row['question']}\nA: {row['answer']}\n\n"

            logger.info(f"🔍 Found {len(similar_data)} relevant context entries")

            # Enhanced prompt for better answers
            prompt = f"""
            You are Career Compass, a career guidance expert. Use the context below to answer the question.

            CONTEXT FROM CAREER DATABASE:
            {context}

            USER QUESTION: {question}

            INSTRUCTIONS:
            - Use the context above as your primary source
            - Provide specific, practical career advice
            - Focus on Lebanese university context when relevant
            - Be concise but comprehensive
            - If context doesn't fully answer, supplement with your knowledge

            ANSWER:
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
                "retrieved_contexts": len(similar_data),
                "confidence": "High",
                "source": "RAG + GPT-4o-mini"
            }

        except Exception as e:
            logger.error(f"❌ RAG error: {e}")
            return self._ask_direct_openai(question)

    def _ask_direct_openai(self, question):
        """Fallback to direct OpenAI"""
        try:
            prompt = f"""
            You are Career Compass, a career guidance expert specializing in Lebanese universities and career paths.
            
            Question: {question}
            
            Provide specific, practical advice about careers, education, and skills development.
            """
            
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )

            return {
                "answer": response.choices[0].message.content.strip(),
                "confidence": "Medium",
                "source": "Direct GPT-4o-mini"
            }
        except Exception as e:
            logger.error(f"Direct OpenAI error: {e}")
            return {
                "answer": "🎓 Career Compass is here to help! I provide expert guidance on careers, education, and skills. What specific questions do you have about career paths or majors?",
                "confidence": "High"
            }

# Create and initialize system
career_system = CareerCompassRAG()
