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
        """Initialize Weaviate Cloud client with v3"""
        try:
            cluster_url = os.getenv("WEAVIATE_CLOUD_URL")
            api_key = os.getenv("WEAVIATE_API_KEY")

            if not cluster_url or not api_key:
                logger.error("❌ Missing Weaviate environment variables")
                return False

            logger.info(f"🔗 Connecting to Weaviate Cloud: {cluster_url}")

            # WEAVIATE V3 CONNECTION (STABLE)
            self.client = weaviate.Client(
                url=cluster_url,
                auth_client_secret=weaviate.AuthApiKey(api_key=api_key),
                additional_headers={
                    "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
                }
            )

            if self.client.is_ready():
                logger.info("✅ Successfully connected to Weaviate Cloud!")
                return True
            else:
                logger.error("❌ Failed to connect to Weaviate Cloud")
                return False

        except Exception as e:
            logger.error(f"❌ Weaviate connection error: {e}")
            return False

    def _check_and_create_schema(self):
        """Check if schema exists, create if not."""
        try:
            class_name = "CareerKnowledge"
            
            if self.client.schema.exists(class_name):
                logger.info("✅ Schema already exists")
                return True
            else:
                logger.info("📋 Creating Weaviate schema...")
                
                class_obj = {
                    "class": class_name,
                    "description": "Career guidance knowledge base",
                    "properties": [
                        {"name": "question", "dataType": ["text"]},
                        {"name": "answer", "dataType": ["text"]},
                        {"name": "is_augmented", "dataType": ["boolean"]},
                        {"name": "source", "dataType": ["text"]}
                    ]
                }
                
                self.client.schema.create_class(class_obj)
                logger.info("✅ Schema created successfully!")
                return True
                
        except Exception as e:
            logger.error(f"❌ Schema creation error: {e}")
            return False

    def _get_embedding_model(self):
        if self.embedding_model is None:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✅ Embedding model loaded")
            except Exception as e:
                logger.error(f"❌ Failed to load embedding model: {e}")
                raise
        return self.embedding_model

    def initialize_system(self, data_path):
        """Initialize the complete RAG system"""
        logger.info("🚀 Initializing Career Compass RAG...")

        # Connect to Weaviate
        if not self._initialize_weaviate_client():
            return False

        # Create schema
        if not self._check_and_create_schema():
            return False

        # Load data
        try:
            if not os.path.exists(data_path):
                logger.error(f"❌ Data file not found: {data_path}")
                return False
                
            df = pd.read_csv(data_path)
            logger.info(f"✅ Loaded {len(df)} rows from CSV")
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            return False

        # Load embedding model
        try:
            embedding_model = self._get_embedding_model()
        except Exception as e:
            logger.error(f"❌ Failed to get embedding model: {e}")
            return False

        # Add documents to Weaviate
        logger.info("💾 Adding documents to Weaviate...")
        try:
            self.client.batch.configure(batch_size=50)
            successful_inserts = 0
            
            with self.client.batch as batch:
                for i, (_, row) in enumerate(df.iterrows()):
                    try:
                        text = str(row.get("answer", ""))
                        if not text.strip():
                            continue
                            
                        embedding = embedding_model.encode(text).tolist()
                        
                        data_object = {
                            "question": str(row.get("question", "")),
                            "answer": text,
                            "is_augmented": False,
                            "source": "career_compass_dataset"
                        }
                        
                        batch.add_data_object(
                            data_object=data_object,
                            class_name="CareerKnowledge",
                            vector=embedding
                        )
                        
                        successful_inserts += 1
                        
                        if (i + 1) % 100 == 0:
                            logger.info(f"📤 Processed {i + 1}/{len(df)} documents")
                            
                    except Exception:
                        continue
            
            logger.info(f"✅ Successfully added {successful_inserts} documents")
            
            if successful_inserts > 0:
                self.is_initialized = True
                logger.info("🎉 Career Compass RAG initialized successfully!")
                return True
            else:
                logger.error("❌ No documents added")
                return False
            
        except Exception as e:
            logger.error(f"❌ Error adding documents: {e}")
            return False

    def ask_question(self, question):
        """Real RAG with OpenAI"""
        try:
            if not self.client or not self.is_initialized:
                return {"answer": "Career guidance system is currently unavailable. Please try again later.", "confidence": "Error"}

            logger.info(f"🤔 Processing question: {question}")

            # Generate embedding for the question
            embedding_model = self._get_embedding_model()
            question_embedding = embedding_model.encode(question).tolist()

            # Search in Weaviate
            response = (
                self.client.query
                .get("CareerKnowledge", ["question", "answer", "source"])
                .with_near_vector({"vector": question_embedding})
                .with_limit(5)
                .do()
            )

            if "data" not in response or not response["data"]["Get"]["CareerKnowledge"]:
                return {"answer": "I don't have enough information to answer that question. Please try asking about career paths, skills, or educational requirements.", "confidence": "Low"}

            chunks = response["data"]["Get"]["CareerKnowledge"]
            context = "\n".join([chunk["answer"] for chunk in chunks])
            logger.info(f"🔍 Retrieved {len(chunks)} relevant chunks")

            # Construct RAG prompt
            prompt = f"""
            You are Career Compass, a helpful career guidance assistant.

            Use the following context to answer the question. If the context doesn't contain relevant information, say so.

            Context:
            {context}

            Question: {question}

            Provide a helpful, career-focused answer based on the context:
            """

            # Call LLM
            llm_response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )

            final_answer = llm_response.choices[0].message.content.strip()
            
            logger.info(f"💡 Generated answer with {len(final_answer)} characters")
            
            return {
                "answer": final_answer,
                "retrieved_chunks": len(chunks),
                "confidence": "High"
            }

        except Exception as e:
            logger.error(f"❌ Error in ask_question: {e}")
            return {"answer": "Sorry, I'm having trouble processing your question right now. Please try again.", "confidence": "Error"}

    def close_connection(self):
        """Close connection"""
        if self.client:
            logger.info("🔌 Connection closed")

if __name__ == "__main__":
    system = CareerCompassWeaviate()
    if system.initialize_system("app/final_merged_career_guidance.csv"):
        response = system.ask_question("What skills are important for AI engineers?")
        print("💡 Answer:", response["answer"])
    system.close_connection()
