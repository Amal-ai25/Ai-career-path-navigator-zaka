import os
import pandas as pd
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import logging
from dotenv import load_dotenv

load_dotenv()

# Set up logging
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

            logger.info(f"🔗 Connecting to Weaviate Cloud: {cluster_url}")

            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=cluster_url,
                auth_credentials=Auth.api_key(api_key)
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
            
            # Check if collection exists
            if self.client.collections.exists(class_name):
                logger.info("✅ Schema already exists")
                return True
            else:
                logger.info("📋 Creating Weaviate schema...")

                self.client.collections.create(
                    name=class_name,
                    properties=[
                        Property(name="question", data_type=DataType.TEXT),
                        Property(name="answer", data_type=DataType.TEXT),
                        Property(name="is_augmented", data_type=DataType.BOOL),
                        Property(name="source", data_type=DataType.TEXT),
                    ]
                )
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
        """Initialize the complete RAG system without langchain-weaviate"""
        logger.info("🚀 Initializing Career Compass RAG...")

        # Step 1: Connect to Weaviate
        if not self._initialize_weaviate_client():
            logger.error("❌ Failed to initialize Weaviate client")
            return False

        # Step 2: Create schema
        if not self._check_and_create_schema():
            logger.error("❌ Failed to create schema")
            return False

        # Step 3: Load data
        logger.info(f"📂 Loading data from: {data_path}")
        try:
            df = pd.read_csv(data_path)
            logger.info(f"📄 Loaded {len(df)} rows from CSV")
        except Exception as e:
            logger.error(f"❌ Failed to load CSV: {e}")
            return False

        # Step 4: Get embedding model
        try:
            embedding_model = self._get_embedding_model()
        except Exception as e:
            logger.error(f"❌ Failed to get embedding model: {e}")
            return False

        # Step 5: Add documents directly to Weaviate
        logger.info("💾 Adding documents to Weaviate...")
        try:
            collection = self.client.collections.get("CareerKnowledge")
            
            batch_size = 20
            successful_inserts = 0
            
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i + batch_size]
                objects = []
                
                for _, row in batch.iterrows():
                    try:
                        text = str(row.get("answer", ""))
                        if not text.strip():
                            continue
                            
                        embedding = embedding_model.encode(text).tolist()
                        
                        objects.append({
                            "properties": {
                                "question": str(row.get("question", "")),
                                "answer": text,
                                "is_augmented": False,
                                "source": "career_compass_dataset"
                            },
                            "vector": embedding
                        })
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to process row {_}: {e}")
                        continue
                
                if objects:
                    collection.data.insert_many(objects)
                    successful_inserts += len(objects)
                    logger.info(f"📤 Added {min(i + batch_size, len(df))}/{len(df)} documents")
            
            logger.info(f"✅ Successfully added {successful_inserts} documents")
            
            if successful_inserts > 0:
                self.is_initialized = True
                logger.info("🎉 Career Compass RAG initialized successfully!")
                return True
            else:
                logger.error("❌ No documents were successfully added")
                return False
            
        except Exception as e:
            logger.error(f"❌ Error adding documents to Weaviate: {e}")
            return False

    def ask_question(self, question):
        """Ask a question and get a synthesized RAG answer"""
        try:
            if not self.client or not self.is_initialized:
                return {
                    "answer": "Career guidance system is currently unavailable. Please try again later.", 
                    "confidence": "Error"
                }

            logger.info(f"🤔 Processing question: {question}")

            # Generate embedding for the question
            embedding_model = self._get_embedding_model()
            question_embedding = embedding_model.encode(question).tolist()

            # Search in Weaviate
            collection = self.client.collections.get("CareerKnowledge")
            
            response = collection.query.near_vector(
                near_vector=question_embedding,
                limit=5,
                return_properties=["question", "answer", "source"]
            )

            if not response.objects:
                return {
                    "answer": "I don't have enough information to answer that question. Please try asking about career paths, skills, or educational requirements.", 
                    "confidence": "Low"
                }

            # Build context
            context = "\n".join([obj.properties["answer"] for obj in response.objects])
            logger.info(f"🔍 Retrieved {len(response.objects)} relevant chunks")

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
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )

            final_answer = response.choices[0].message.content.strip()
            
            logger.info(f"💡 Generated answer with {len(final_answer)} characters")
            
            return {
                "answer": final_answer,
                "retrieved_chunks": len(response.objects),
                "confidence": "High"
            }

        except Exception as e:
            logger.error(f"❌ Error in ask_question: {e}")
            return {
                "answer": "Sorry, I'm having trouble processing your question right now. Please try again.", 
                "confidence": "Error"
            }

    def close_connection(self):
        """Close the connection."""
        logger.info("🔌 Closing Weaviate connection...")
        if self.client:
            self.client.close()
        logger.info("✅ Connection closed")

if __name__ == "__main__":
    system = CareerCompassWeaviate()
    if system.initialize_system("app/final_merged_career_guidance.csv"):
        response = system.ask_question("What skills are important for AI engineers?")
        print("💡 Answer:", response["answer"])
    system.close_connection()
