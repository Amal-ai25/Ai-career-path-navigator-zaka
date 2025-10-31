import os
import pandas as pd
import weaviate
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CareerCompassWeaviate:
    def __init__(self):
        self.client = None
        self.embedding_model = None
        self.llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.is_initialized = False

    def _initialize_weaviate_client(self):
        """Initialize Weaviate Cloud client"""
        try:
            cluster_url = os.getenv("WEAVIATE_CLOUD_URL")
            api_key = os.getenv("WEAVIATE_API_KEY")

            if not cluster_url or not api_key:
                logger.error("❌ Missing Weaviate environment variables")
                return False

            logger.info("🔗 Connecting to Weaviate Cloud...")

            # Simple Weaviate connection
            self.client = weaviate.Client(
                url=cluster_url,
                auth_client_secret=weaviate.AuthApiKey(api_key=api_key)
            )

            if self.client.is_ready():
                logger.info("✅ Connected to Weaviate Cloud!")
                return True
            else:
                logger.error("❌ Weaviate connection failed")
                return False

        except Exception as e:
            logger.error(f"❌ Weaviate connection error: {e}")
            return False

    def initialize_system(self, data_path):
        """Initialize RAG system - simplified version"""
        logger.info("🚀 Starting Career Compass...")

        # Try to connect to Weaviate
        weaviate_connected = self._initialize_weaviate_client()
        
        if not weaviate_connected:
            logger.warning("⚠️ Weaviate not available, using fallback mode")
            # Still mark as initialized for fallback functionality
            self.is_initialized = True
            return True

        # Load embedding model
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Embedding model loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            return False

        self.is_initialized = True
        logger.info("✅ Career Compass initialized (Weaviate: Connected)")
        return True

    def ask_question(self, question):
        """Ask question with fallback to direct OpenAI"""
        try:
            if not self.is_initialized:
                return {
                    "answer": "Career guidance system is starting up. Please try again in a moment.",
                    "confidence": "Low"
                }

            # If Weaviate is connected, try RAG
            if self.client and self.client.is_ready():
                try:
                    # Simple search in Weaviate
                    response = (
                        self.client.query
                        .get("CareerKnowledge", ["question", "answer"])
                        .with_limit(3)
                        .do()
                    )

                    if response.get("data", {}).get("Get", {}).get("CareerKnowledge"):
                        context = "\n".join([
                            chunk["answer"] for chunk in response["data"]["Get"]["CareerKnowledge"]
                        ])
                        
                        prompt = f"""
                        Based on this career guidance context, answer the question.

                        Context:
                        {context}

                        Question: {question}

                        Answer helpfully:
                        """
                    else:
                        prompt = f"Answer this career guidance question: {question}"
                        
                except Exception as e:
                    logger.warning(f"⚠️ Weaviate search failed, using direct OpenAI: {e}")
                    prompt = f"Answer this career guidance question: {question}"
            else:
                # Direct OpenAI fallback
                prompt = f"""
                You are Career Compass, a career guidance assistant. 
                Answer this question about careers, education, skills, or majors:

                {question}
                """

            # Generate answer
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )

            answer = response.choices[0].message.content.strip()
            
            return {
                "answer": answer,
                "confidence": "High",
                "source": "OpenAI" if not self.client else "Weaviate+RAG"
            }

        except Exception as e:
            logger.error(f"❌ Error in ask_question: {e}")
            return {
                "answer": "I'm here to help with career guidance! Please try asking about different career paths, required skills, or educational programs.",
                "confidence": "Medium"
            }

# Create instance immediately
career_system = CareerCompassWeaviate()
