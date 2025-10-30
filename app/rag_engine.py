import os
import pandas as pd
import weaviate
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
        """Initialize connection to Weaviate Cloud"""
        try:
            cluster_url = os.getenv("WEAVIATE_CLOUD_URL")
            api_key = os.getenv("WEAVIATE_API_KEY")

            if not cluster_url or not api_key:
                logger.error("❌ Missing Weaviate environment variables")
                return False

            logger.info(f"🔗 Connecting to Weaviate Cloud: {cluster_url}")
            
            # CORRECT Weaviate v3 connection method
            self.client = weaviate.Client(
                url=cluster_url,
                auth_client_secret=weaviate.AuthApiKey(api_key=api_key),
                additional_headers={
                    "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
                }
            )

            # Test connection
            if self.client.is_ready():
                logger.info("✅ Connected to Weaviate Cloud successfully!")
                return True
            else:
                logger.error("❌ Weaviate not ready")
                return False
                
        except Exception as e:
            logger.error(f"❌ Weaviate connection error: {e}")
            return False

    def _check_and_create_schema(self):
        """Ensure the schema exists"""
        try:
            class_name = "CareerKnowledge"
            
            # Check if class exists
            if self.client.schema.exists(class_name):
                logger.info("✅ Schema already exists")
                return True
            else:
                logger.info("📋 Creating Weaviate schema...")
                
                class_obj = {
                    "class": class_name,
                    "description": "Career guidance knowledge base",
                    "properties": [
                        {
                            "name": "question",
                            "dataType": ["text"],
                            "description": "The question"
                        },
                        {
                            "name": "answer", 
                            "dataType": ["text"],
                            "description": "The answer"
                        },
                        {
                            "name": "is_augmented",
                            "dataType": ["boolean"],
                            "description": "Whether the data was augmented"
                        },
                        {
                            "name": "source",
                            "dataType": ["text"],
                            "description": "Source of the data"
                        }
                    ]
                }
                
                self.client.schema.create_class(class_obj)
                logger.info("✅ Schema created successfully")
                return True
                
        except Exception as e:
            logger.error(f"❌ Schema creation error: {e}")
            return False

    def _get_embedding_model(self):
        if self.embedding_model is None:
            try:
                logger.info("🔄 Loading embedding model...")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✅ Embedding model loaded")
            except Exception as e:
                logger.error(f"❌ Failed to load embedding model: {e}")
                raise
        return self.embedding_model

    def initialize_system(self, data_path):
        """Initialize the complete RAG system"""
        logger.info(f"🚀 Initializing Career Compass RAG with data: {data_path}")

        # Step 1: Connect to Weaviate
        if not self._initialize_weaviate_client():
            logger.error("❌ Weaviate client initialization failed")
            return False

        # Step 2: Create schema
        if not self._check_and_create_schema():
            logger.error("❌ Schema creation failed")
            return False

        # Step 3: Load data
        try:
            if not os.path.exists(data_path):
                logger.error(f"❌ Data file not found: {data_path}")
                return False
                
            logger.info("📄 Loading CSV data...")
            df = pd.read_csv(data_path)
            logger.info(f"✅ Loaded {len(df)} rows from CSV")
            
            # Check required columns
            required_columns = ['question', 'answer']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"❌ Missing required columns: {missing_columns}")
                logger.info(f"📋 Available columns: {list(df.columns)}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            return False

        # Step 4: Load embedding model
        try:
            embedding_model = self._get_embedding_model()
        except Exception as e:
            logger.error(f"❌ Failed to get embedding model: {e}")
            return False

        # Step 5: Import data to Weaviate
        logger.info("💾 Adding documents to Weaviate...")
        try:
            batch_size = 20
            successful_inserts = 0
            
            # Configure batch
            self.client.batch.configure(batch_size=batch_size)
            
            with self.client.batch as batch:
                for i, (_, row) in enumerate(df.iterrows()):
                    try:
                        text = str(row.get("answer", ""))
                        if not text.strip():
                            continue
                            
                        # Generate embedding
                        embedding = embedding_model.encode(text).tolist()
                        
                        # Create data object
                        data_object = {
                            "question": str(row.get("question", "")),
                            "answer": text,
                            "is_augmented": False,
                            "source": "career_compass_dataset"
                        }
                        
                        # Add to batch
                        batch.add_data_object(
                            data_object=data_object,
                            class_name="CareerKnowledge",
                            vector=embedding
                        )
                        
                        successful_inserts += 1
                        
                        if (i + 1) % 100 == 0:
                            logger.info(f"📤 Processed {i + 1}/{len(df)} documents")
                            
                    except Exception as e:
                        if i < 5:  # Log first few errors
                            logger.warning(f"⚠️ Failed to process row {i}: {e}")
                        continue
            
            logger.info(f"✅ Successfully added {successful_inserts}/{len(df)} documents")
            
            if successful_inserts > 0:
                self.is_initialized = True
                logger.info("🎉 Career Compass RAG system is fully initialized and ready!")
                return True
            else:
                logger.error("❌ No documents were successfully added")
                return False
            
        except Exception as e:
            logger.error(f"❌ Error adding documents to Weaviate: {e}")
            return False

    def ask_question(self, question):
        """Retrieve + Generate an answer using RAG"""
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
            logger.info(f"🔍 Retrieved {len(chunks)} relevant chunks")
            
            context = "\n".join([chunk["answer"] for chunk in chunks])

            prompt = f"""
            You are Career Compass, a career guidance assistant.

            Use the context below to answer the question. If the context doesn't contain relevant information, say so.

            Context:
            {context}

            Question: {question}

            Provide a helpful, career-focused answer based on the context:
            """

            # Generate answer using OpenAI
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
        """Close connection to Weaviate"""
        logger.info("🔌 Closing Weaviate connection")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "final_merged_career_guidance.csv")

    logger.info(f"📁 Looking for dataset at: {csv_path}")

    system = CareerCompassWeaviate()
    if system.initialize_system(csv_path):
        response = system.ask_question("What skills are important for AI engineers?")
        logger.info(f"💡 Answer: {response['answer']}")
    system.close_connection()
