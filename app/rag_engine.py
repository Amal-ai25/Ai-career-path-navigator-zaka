import os
import pandas as pd
from openai import OpenAI
import logging
from dotenv import load_dotenv
import re

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CareerCompassRAG:
    def __init__(self):
        self.llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.career_data = None
        self.is_initialized = False
        logger.info("✅ Career RAG system initializing")

    def _load_career_data(self, data_path):
        """Load career data without embeddings"""
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
        """Initialize the simple RAG system"""
        logger.info("🚀 Initializing Simple Career RAG...")
        
        if self._load_career_data(data_path):
            self.is_initialized = True
            logger.info("🎉 Career RAG initialized successfully!")
            return True
        return False

    def _find_relevant_qa(self, question, top_k=5):
        """Find relevant Q&A using simple keyword matching"""
        if self.career_data is None:
            return []
        
        question_lower = question.lower()
        question_words = set(re.findall(r'\b\w+\b', question_lower))
        
        matches = []
        
        for idx, row in self.career_data.iterrows():
            if pd.notna(row['question']) and pd.notna(row['answer']):
                q_text = str(row['question']).lower()
                a_text = str(row['answer']).lower()
                
                # Calculate simple match score
                q_match = len(question_words.intersection(set(re.findall(r'\b\w+\b', q_text))))
                a_match = len(question_words.intersection(set(re.findall(r'\b\w+\b', a_text))))
                
                total_score = q_match * 3 + a_match  # Weight question matches higher
                
                if total_score > 0:
                    matches.append((total_score, idx, row))
        
        # Sort by score and return top matches
        matches.sort(reverse=True, key=lambda x: x[0])
        return [match[2] for match in matches[:top_k]]

    def ask_question(self, question):
        """Ask question with simple keyword-based RAG"""
        try:
            if not self.is_initialized or self.career_data is None:
                return self._ask_direct_openai(question)

            logger.info(f"🤔 Processing: {question}")

            # Find relevant Q&A using keyword matching
            relevant_data = self._find_relevant_qa(question)
            
            if len(relevant_data) == 0:
                # No relevant data found, use direct OpenAI with career context
                return self._ask_career_openai(question)

            # Build context from relevant questions
            context = "RELEVANT CAREER KNOWLEDGE:\n"
            for row in relevant_data:
                context += f"Q: {row['question']}\nA: {row['answer']}\n\n"

            logger.info(f"🔍 Found {len(relevant_data)} relevant Q&A pairs")

            # Smart prompt for better answers
            prompt = f"""
            You are Career Compass, an expert career guidance assistant. 
            Use the career knowledge below to answer the user's question.

            {context}

            USER QUESTION: {question}

            INSTRUCTIONS:
            1. Use the career knowledge above as your primary source
            2. Provide specific, practical advice based on the context
            3. If the context doesn't fully answer, supplement with your career expertise
            4. Focus on actionable guidance and clear explanations
            5. Keep answers comprehensive but concise

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
                "relevant_matches": len(relevant_data),
                "confidence": "High",
                "source": "Career Database + GPT-4o-mini"
            }

        except Exception as e:
            logger.error(f"❌ RAG error: {e}")
            return self._ask_direct_openai(question)

    def _ask_career_openai(self, question):
        """OpenAI with career expert context"""
        try:
            prompt = f"""
            You are Career Compass, a career guidance expert specializing in:
            - Career paths and job markets
            - University majors and education
            - Skill development and certifications
            - Lebanese university context
            - Professional development advice

            Question: {question}

            Provide specific, practical career guidance based on your expertise.
            """
            
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=350
            )

            return {
                "answer": response.choices[0].message.content.strip(),
                "confidence": "Medium",
                "source": "Career Expert GPT-4o-mini"
            }
        except Exception as e:
            logger.error(f"Direct OpenAI error: {e}")
            return self._get_fallback_response(question)

    def _ask_direct_openai(self, question):
        """Simple direct OpenAI fallback"""
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": question}],
                temperature=0.3,
                max_tokens=300
            )

            return {
                "answer": response.choices[0].message.content.strip(),
                "confidence": "Medium", 
                "source": "Direct GPT-4o-mini"
            }
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return self._get_fallback_response(question)

    def _get_fallback_response(self, question):
        """Intelligent fallback responses"""
        question_lower = question.lower()
        
        fallbacks = {
            'law': "Law degrees offer specializations like Criminal Law, Corporate Law, Family Law, Intellectual Property, and International Law. Each has different career paths and requirements.",
            'computer': "Computer Science leads to careers in software development, AI, data science, cybersecurity, and more. Key skills include programming, algorithms, and problem-solving.",
            'business': "Business majors can specialize in Management, Marketing, Finance, or Entrepreneurship. These lead to diverse careers in various industries.",
            'engineering': "Engineering fields include Civil, Mechanical, Electrical, Computer, and Chemical Engineering. Each requires specific technical skills and offers different career opportunities.",
            'medicine': "Medical careers include Doctors, Surgeons, Dentists, Pharmacists, and Healthcare Administrators. These require extensive education and specialized training.",
            'skill': "Important career skills include communication, problem-solving, technical expertise, adaptability, and leadership. Continuous learning is key for career growth.",
            'career': "Consider your interests, skills, job market demand, and education requirements when choosing a career path. Research different fields and talk to professionals.",
            'major': "Popular majors include Computer Science, Business, Engineering, Healthcare, and Social Sciences. Choose based on your interests and career goals.",
            'lebanese': "Lebanese universities offer quality education in various fields. Consider factors like accreditation, faculty, facilities, and career support when choosing.",
            'university': "When selecting a university, consider program quality, location, cost, career services, and alumni network. Research thoroughly before deciding."
        }
        
        for keyword, response in fallbacks.items():
            if keyword in question_lower:
                return {
                    "answer": response,
                    "confidence": "High",
                    "source": "Career Knowledge Base"
                }
        
        return {
            "answer": "🎓 Career Compass is here to help! I provide expert guidance on careers, education, majors, and skills. What specific career questions can I assist with?",
            "confidence": "High",
            "source": "Career Assistant"
        }

# Create and initialize system
career_system = CareerCompassRAG()
