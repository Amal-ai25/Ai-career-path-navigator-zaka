import os
import pandas as pd
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
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
        self.vectorstore = None
        self._llm_client = None
    
    def _get_llm_client(self):
        """Lazy initialization of OpenAI client"""
        if self._llm_client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            self._llm_client = OpenAI(api_key=api_key)
        return self._llm_client
    
    # --- 1️⃣ Connect to Weaviate Cloud ---
    def _initialize_weaviate_client(self):
        """Initialize connection to Weaviate Cloud"""
        try:
            cluster_url = os.getenv("WEAVIATE_CLOUD_URL")
            api_key = os.getenv("WEAVIATE_API_KEY")

            logger.info(f"🔗 Connecting to Weaviate Cloud: {cluster_url}")
            logger.info(f"📂 Current working directory: {os.getcwd()}")
            
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

    # --- 2️⃣ Schema Management ---
    def _check_and_create_schema(self):
        """Ensure the schema (CareerKnowledge) exists"""
        try:
            class_name = "CareerKnowledge"
            schema = self.client.collections.list_all()

            if class_name not in schema:
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
            else:
                logger.info("✅ Schema already exists")

            return True
        except Exception as e:
            logger.error(f"❌ Schema creation error: {e}")
            return False

    # --- 3️⃣ Initialize Data + Embeddings ---
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

        # Chunk text
        logger.info("✂️ Splitting text into token chunks...")
        text_splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=20)

        documents = []
        for _, row in df.iterrows():
            chunks = text_splitter.split_text(row["answer"])
            for chunk in chunks:
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "question": row["question"],
                        "answer": row["answer"],
                        "is_augmented": False,
                        "source": "career_compass_dataset"
                    }
                )
                documents.append(doc)

        logger.info(f"📝 Prepared {len(documents)} chunks")

        # Embeddings
        logger.info("🧠 Initializing embedding model...")
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Vector store
        logger.info("💾 Creating vector store in Weaviate...")
        self.vectorstore = WeaviateVectorStore(
            client=self.client,
            index_name="CareerKnowledge",
            text_key="answer",
            embedding=embedding_model,
            attributes=["question", "answer", "is_augmented", "source"]
        )

        # Add documents
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            self.vectorstore.add_documents(documents[i:i + batch_size])
            logger.info(f"📤 Added {min(i + batch_size, len(documents))}/{len(documents)}")

        logger.info("✅ All documents added successfully")
        logger.info("🎉 Career Compass RAG is ready!")
        return True

    # --- 4️⃣ Ask a Question ---
    def ask_question(self, question):
        """Retrieve + Generate an answer using RAG"""
        try:
            if not self.vectorstore:
                return {"answer": "System not initialized.", "confidence": "Error"}

            results = self.vectorstore.similarity_search(query=question, k=5)

            if not results:
                return {"answer": "I don't have enough information.", "confidence": "Low"}

            context = "\n".join([doc.page_content for doc in results])

            prompt = f"""
            You are Career Compass, a career guidance assistant.

            Use the context below to answer the question.

            Context:
            {context}

            Question: {question}
            Answer:
            """

            llm_client = self._get_llm_client()
            response = llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )

            final_answer = response.choices[0].message.content.strip()
            return {
                "answer": final_answer,
                "retrieved_chunks": len(results),
                "confidence": "High"
            }

        except Exception as e:
            logger.error(f"❌ Error in ask_question: {e}")
            return {"answer": f"Error: {e}", "confidence": "Error"}

    # --- 5️⃣ Cleanup ---
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
    system.initialize_system(csv_path)
    response = system.ask_question("What skills are important for AI engineers?")
    logger.info(f"💡 Answer: {response['answer']}")
    system.close_connection()
