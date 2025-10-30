import os
import pandas as pd
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType
from langchain_community.embeddings import HuggingFaceEmbeddings
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

    def initialize_system(self, data_path):
        """Initialize the complete RAG system."""
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

        # Step 4: Token-based chunking (like your Colab)
        logger.info("✂️ Splitting text into token-based chunks...")
        text_splitter = TokenTextSplitter(
            chunk_size=200,    # 200 tokens per chunk
            chunk_overlap=20
        )

        documents = []
        for _, row in df.iterrows():
            try:
                answer_text = str(row["answer"]) if pd.notna(row["answer"]) else ""
                question_text = str(row["question"]) if pd.notna(row["question"]) else ""
                
                if not answer_text.strip():
                    continue
                    
                chunks = text_splitter.split_text(answer_text)
                for chunk in chunks:
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "question": question_text,
                            "answer": answer_text,
                            "is_augmented": False,
                            "source": "career_compass_dataset"
                        }
                    )
                    documents.append(doc)
            except Exception as e:
                logger.warning(f"⚠️ Failed to process row {_}: {e}")
                continue

        logger.info(f"📝 Prepared {len(documents)} chunks for embedding")

        # Step 5: Initialize embeddings (like your Colab)
        logger.info("🧠 Initializing embeddings...")
        try:
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("✅ Embeddings initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize embeddings: {e}")
            return False

        # Step 6: Create vector store (like your Colab)
        logger.info("💾 Creating vector store...")
        try:
            self.vectorstore = WeaviateVectorStore(
                client=self.client,
                index_name="CareerKnowledge",
                text_key="answer",
                embedding=embedding_model,
                attributes=["question", "answer", "is_augmented", "source"]
            )
            logger.info("✅ Vector store created")
        except Exception as e:
            logger.error(f"❌ Failed to create vector store: {e}")
            return False

        # Step 7: Add documents in batches
        logger.info("📤 Adding documents to Weaviate...")
        try:
            batch_size = 100
            total_documents = len(documents)
            
            for i in range(0, total_documents, batch_size):
                batch = documents[i:i + batch_size]
                self.vectorstore.add_documents(documents=batch)
                
                if (i // batch_size) % 10 == 0 or i + batch_size >= total_documents:
                    logger.info(f"   Added {min(i + batch_size, total_documents)}/{total_documents} chunks")

            logger.info(f"✅ Added {total_documents} chunks to Weaviate Cloud")
            self.is_initialized = True
            logger.info("🎉 Career Compass RAG initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding documents to Weaviate: {e}")
            return False

    def ask_question(self, question):
        """Ask a question and get a synthesized RAG answer (like your Colab)."""
        try:
            if not self.vectorstore or not self.is_initialized:
                return {
                    "answer": "Career guidance system is currently unavailable. Please try again later.", 
                    "confidence": "Error"
                }

            logger.info(f"🤔 Processing question: {question}")

            # Step 1: Retrieve relevant chunks (like your Colab)
            results = self.vectorstore.similarity_search(query=question, k=5)

            if not results:
                return {
                    "answer": "I don't have enough information to answer that question. Please try asking about career paths, skills, or educational requirements.", 
                    "confidence": "Low"
                }

            # Step 2: Build context (like your Colab)
            context = "\n".join([doc.page_content for doc in results])
            logger.info(f"🔍 Retrieved {len(results)} relevant chunks")

            # Step 3: Construct RAG prompt (like your Colab)
            prompt = f"""
            You are Career Compass, a helpful career guidance assistant.

            Use the following context to answer the question. If the context doesn't contain relevant information, say so.

            Context:
            {context}

            Question: {question}

            Provide a helpful, career-focused answer based on the context:
            """

            # Step 4: Call LLM (like your Colab)
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
                "retrieved_chunks": len(results),
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

# Test function (like your Colab)
def test_rag_system():
    """Test the RAG system with sample questions"""
    system = CareerCompassWeaviate()
    
    # Try multiple dataset paths
    dataset_paths = [
        "app/final_merged_career_guidance.csv",
        "./app/final_merged_career_guidance.csv", 
        "final_merged_career_guidance.csv"
    ]
    
    initialized = False
    for path in dataset_paths:
        if os.path.exists(path):
            logger.info(f"📁 Testing with dataset: {path}")
            if system.initialize_system(path):
                initialized = True
                break
    
    if not initialized:
        logger.error("❌ Could not initialize RAG system with any dataset path")
        return
    
    # Test questions (like your Colab)
    test_questions = [
        "What skills are important for AI engineers?",
        "What are the main areas of specialization in a law degree",
        "What skills are important to succeed as a law student?",
        "How long does it typically take to complete a law degree in lebanon?",
        "What courses are essential in a business management degree?",
        "what is the best universities in lebanon"
    ]
    
    for question in test_questions:
        try:
            response = system.ask_question(question)
            logger.info(f"💡 Q: {question}")
            logger.info(f"💡 A: {response['answer']}")
            logger.info("---" * 20)
        except Exception as e:
            logger.error(f"❌ Error testing question '{question}': {e}")
    
    system.close_connection()

if __name__ == "__main__":
    test_rag_system()
