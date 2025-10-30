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
    
    def _initialize_weaviate_client(self):
        """Initialize connection to Weaviate Cloud"""
        try:
            cluster_url = os.getenv("WEAVIATE_CLOUD_URL")
            api_key = os.getenv("WEAVIATE_API_KEY")

            logger.info(f"🔗 Connecting to Weaviate Cloud: {cluster_url}")
            
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=cluster_url,
                auth_credentials=Auth.api_key(api_key)
            )

            if self.client.is_ready():
                logger.info("✅ Connected to Weaviate Cloud")
                return True
            else:
                logger.error("❌ Weaviate not ready")
                return False
        except Exception as e:
            logger.error(f"❌ Connection error: {e}")
            return False

    def _check_and_create_schema(self):
        """Ensure the schema (CareerKnowledge) exists"""
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
                logger.info("✅ Schema created")
                return True
        except Exception as e:
            logger.error(f"❌ Schema creation error: {e}")
            return False

    def _get_embedding_model(self):
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self.embedding_model

    def initialize_system(self, data_path):
        """Initialize the complete RAG system"""
        logger.info("🚀 Initializing Career Compass RAG...")

        if not self._initialize_weaviate_client():
            logger.error("❌ Weaviate client init failed")
            return False

        if not self._check_and_create_schema():
            logger.error("❌ Schema creation failed")
            return False

        try:
            if not os.path.exists(data_path):
                logger.error(f"❌ Data file not found: {data_path}")
                return False
                
            df = pd.read_csv(data_path)
            logger.info(f"📄 Loaded {len(df)} rows from CSV")
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            return False

        # Get embedding model
        embedding_model = self._get_embedding_model()

        # Add documents directly to Weaviate
        logger.info("💾 Adding documents to Weaviate...")
        try:
            collection = self.client.collections.get("CareerKnowledge")
            
            batch_size = 20
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i + batch_size]
                objects = []
                
                for _, row in batch.iterrows():
                    # Generate embedding for the full answer
                    text = str(row.get("answer", ""))
                    embedding = embedding_model.encode(text).tolist()
                    
                    # Create object with embedding
                    objects.append({
                        "properties": {
                            "question": str(row.get("question", "")),
                            "answer": text,
                            "is_augmented": False,
                            "source": "career_compass_dataset"
                        },
                        "vector": embedding
                    })
                
                # Insert batch with embeddings
                collection.data.insert_many(objects)
                logger.info(f"📤 Added {min(i + batch_size, len(df))}/{len(df)} documents")
            
            logger.info("✅ All documents added successfully")
            logger.info("🎉 Career Compass RAG is ready!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding documents: {e}")
            return False

    def ask_question(self, question):
        """Retrieve + Generate an answer using RAG"""
        try:
            if not self.client:
                return {"answer": "System not initialized.", "confidence": "Error"}

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
                return {"answer": "I don't have enough information.", "confidence": "Low"}

            context = "\n".join([obj.properties["answer"] for obj in response.objects])

            prompt = f"""
            You are Career Compass, a career guidance assistant.

            Use the context below to answer the question.

            Context:
            {context}

            Question: {question}
            Answer:
            """

            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )

            final_answer = response.choices[0].message.content.strip()
            return {
                "answer": final_answer,
                "retrieved_chunks": len(response.objects),
                "confidence": "High"
            }

        except Exception as e:
            logger.error(f"❌ Error in ask_question: {e}")
            return {"answer": f"Error: {e}", "confidence": "Error"}

    def close_connection(self):
        """Close connection to Weaviate"""
        if self.client:
            self.client.close()
            logger.info("🔌 Connection closed.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "final_merged_career_guidance.csv")

    logger.info(f"📁 Looking for dataset at: {csv_path}")

    system = CareerCompassWeaviate()
    if system.initialize_system(csv_path):
        response = system.ask_question("What skills are important for AI engineers?")
        logger.info(f"💡 Answer: {response['answer']}")
    system.close_connection()
