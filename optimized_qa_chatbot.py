import os
import re
import logging
from typing import List, Dict
from dataclasses import dataclass
from collections import deque
import time

from google import genai
from pinecone import Pinecone
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration class for the chatbot"""
    pinecone_api_key: str
    gemini_api_key: str
    index_name: str = "qa-chatbot"
    namespace: str = "bizz-docs-sample"
    max_context_length: int = 10  # Limit context cache
    max_history_length: int = 5   # Limit conversation history
    top_k_results: int = 5
    region: str = "us-east-1"
    cloud: str = "aws"

class DocumentManager:
    """Manages document operations and caching"""
    
    def __init__(self, config: Config):
        self.config = config
        self.pc = Pinecone(api_key=config.pinecone_api_key)
        self.index = None
        self._context_cache = deque(maxlen=config.max_context_length)
        
    def initialize_index(self) -> bool:
        """Initialize Pinecone index with error handling"""
        try:
            if not self.pc.has_index(self.config.index_name):
                logger.info(f"Creating index: {self.config.index_name}")
                self.pc.create_index_for_model(
                    name=self.config.index_name,
                    cloud=self.config.cloud,
                    region=self.config.region,
                    embed={
                        "model": "llama-text-embed-v2",
                        "field_map": {"text": "content"}
                    }
                )
                # Wait for index to be ready
                time.sleep(4)
            
            # Get index host dynamically
            index_description = self.pc.describe_index(self.config.index_name)
            host = index_description.host
            self.index = self.pc.Index(host=host)
            logger.info("Index initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize index: {e}")
            return False
    
    def upload_documents(self, documents: List[Dict]) -> bool:
        """Upload documents with duplicate checking"""
        try:
            if not self.index:
                logger.error("Index not initialized")
                return False
                
            # Check if documents already exist
            existing_ids = set()
            try:
                stats = self.index.describe_index_stats()
                if self.config.namespace in stats.namespaces:
                    existing_ids = set(stats.namespaces[self.config.namespace].vector_count)
            except:
                pass
            
            # Filter out existing documents
            new_documents = [doc for doc in documents if doc["_id"] not in existing_ids]
            
            if new_documents:
                logger.info(f"Uploading {len(new_documents)} new documents")
                self.index.upsert_records(self.config.namespace, new_documents)
            else:
                logger.info("All documents already exist in index")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload documents: {e}")
            return False
    
    def retrieve_relevant_docs(self, query: str) -> List[Dict]:
        """Retrieve relevant documents with caching"""
        try:
            search_query = {
                "inputs": {"text": query},
                "top_k": self.config.top_k_results
            }
            
            results = self.index.search(
                namespace=self.config.namespace,
                query=search_query,
                fields=["content", "category", "product_name"]
            )
            
            retrieved_docs = []
            for hit in results['result']['hits']:
                doc = {
                    "id": hit['_id'],
                    "category": hit['fields']['category'],
                    "product_name": hit['fields'].get('product_name', ''),
                    "content": hit['fields']['content'],
                    "score": hit['score']
                }
                retrieved_docs.append(doc)
            
            # Add to cache and return
            self._context_cache.extend(retrieved_docs)
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            return []
    
    def get_context_cache(self) -> List[Dict]:
        """Get current context cache"""
        return list(self._context_cache)

class LLMClient:
    """Manages LLM interactions"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = genai.Client()
        
    def generate_response(self, prompt: str) -> str:
        """Generate response with error handling and retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )
                return self._format_response(response.text)
                
            except Exception as e:
                logger.warning(f"LLM request failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return "I apologize, but I'm having trouble generating a response right now. Please try again."
                time.sleep(1)  # Brief delay before retry
    
    def _format_response(self, response: str) -> str:
        """Format LLM response"""
        # Remove gemini formatting and convert to plain text
        clean_text = re.sub(r"\*\*(.*?)\*\*", r"\1", response)
        clean_text = re.sub(r"^\s*[\*\-]\s*", "- ", clean_text, flags=re.MULTILINE)
        clean_text = re.sub(r'\n{2,}', '\n\n', clean_text)
        return clean_text.strip()

class PromptBuilder:
    """Builds prompts for the LLM"""
    
    @staticmethod
    def build_qa_prompt(user_message: str, context_docs: List[Dict], 
                       conversation_history: List[Dict]) -> str:
        """Build a well-structured prompt for QA"""
        
        # Format context documents
        context_text = "\n\n".join([
            f"Product: {doc.get('product_name', 'N/A')}\nCategory: {doc['category']}\nContent: {doc['content']}"
            for doc in context_docs
        ])
        
        # Format conversation history
        history_context = "\n".join([
            f"{turn['role'].capitalize()}: {turn['content']}" 
            for turn in conversation_history[-4:]
        ])
        
        prompt = f"""You are a Business Domain QA Chatbot powered by RAG (Retrieval-Augmented Generation).

                Your task is to answer questions based on the provided business context. Follow these guidelines:
                - Answer based ONLY on the provided context
                - If the context doesn't contain enough information, say "I don't have enough information to answer that question."
                - Be concise but informative
                - Do not engage in casual conversation or off-topic discussions
                - If asked about multiple products, provide information about each one

                Context Information:
                {context_text}

                Previous Conversation:
                {history_context if history_context else "No previous conversation"}

                Current Question: {user_message}

                Answer:"""
        
        return prompt

class QAChatbot:
    """Main chatbot class with optimized architecture"""
    
    def __init__(self, config: Config):
        self.config = config
        self.doc_manager = DocumentManager(config)
        self.llm_client = LLMClient(config)
        self.prompt_builder = PromptBuilder()
        self.conversation_history = deque(maxlen=config.max_history_length)
        
    def initialize(self, documents: List[Dict]) -> bool:
        """Initialize the chatbot"""
        logger.info("Initializing QA Chatbot...")
        
        if not self.doc_manager.initialize_index():
            return False
            
        if not self.doc_manager.upload_documents(documents):
            return False
            
        logger.info("Chatbot initialized successfully")
        return True
    
    def process_query(self, user_message: str) -> str:
        """Process a user query and return response"""
        try:
            # Add user message to history
            self.conversation_history.append({"role": "user", "content": user_message})
            
            # Retrieve relevant documents
            relevant_docs = self.doc_manager.retrieve_relevant_docs(user_message)
            
            if not relevant_docs:
                return "I don't have enough information to answer that question."
            
            # Build prompt
            prompt = self.prompt_builder.build_qa_prompt(
                user_message, 
                relevant_docs, 
                list(self.conversation_history)
            )
            
            # Generate response
            response = self.llm_client.generate_response(prompt)
            
            # Add bot response to history
            self.conversation_history.append({"role": "bot", "content": response})
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return "I encountered an error while processing your request. Please try again."
    
    def run_interactive(self):
        """Run the chatbot in interactive mode"""
        print("Welcome to the Business QA Chatbot!")
        print("Type 'exit' or 'quit' to end the session.")
        print("Type 'clear' to clear conversation history.")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['exit', 'quit']:
                    print("Goodbye!")
                    break
                    
                if user_input.lower() == 'clear':
                    self.conversation_history.clear()
                    print("Conversation history cleared.")
                    continue
                
                if not user_input:
                    continue
                
                print("\nBot: ", end="")
                response = self.process_query(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                print("An unexpected error occurred. Please try again.")

def load_config() -> Config:
    """Load configuration from environment variables"""
    load_dotenv()
    
    pinecone_key = os.getenv('PINECONE_API_KEY')
    gemini_key = os.getenv('GEMINI_API_KEY')
    
    if not pinecone_key or not gemini_key:
        raise ValueError("Missing required environment variables: PINECONE_API_KEY, GEMINI_API_KEY")
    
    return Config(
        pinecone_api_key=pinecone_key,
        gemini_api_key=gemini_key
    )

# Business documents data
business_documents = [
    # OPENAI DOCS
    {
        "_id": "openai_doc1",
        "content": "ChatGPT is OpenAI's conversational AI assistant, capable of engaging in natural, context-aware dialogue across a wide range of topics. It is used for tasks like writing assistance, tutoring, customer support, and ideation. ChatGPT supports multimodal input (text and images), code generation, and advanced memory features for personalized experiences.",
        "category": "product_info",
        "product_name": "ChatGPT"
    },
    {
        "_id": "openai_doc2",
        "content": "OpenAI Codex is the engine behind GitHub Copilot and is designed to translate natural language into code. It supports over a dozen programming languages and enables developers to build applications faster by generating code snippets, completing functions, and even writing entire modules based on descriptive input.",
        "category": "product_info",
        "product_name": "Codex"
    },
    {
        "_id": "openai_doc3",
        "content": "DALL·E is a generative AI model by OpenAI that creates images from textual prompts. With support for inpainting, editing, and realistic rendering, DALL·E empowers artists, marketers, and developers to turn creative ideas into visual content. It's particularly popular for concept art, design mockups, and marketing illustrations.",
        "category": "product_info",
        "product_name": "DALL·E"
    },
    {
        "_id": "openai_doc4",
        "content": "Whisper is OpenAI's automatic speech recognition (ASR) system trained on a large, diverse dataset of multilingual and multitask audio. It delivers high-accuracy transcription, translation, and speech-to-text capabilities. Whisper is open-source and used in voice assistants, accessibility apps, and real-time transcription tools.",
        "category": "product_info",
        "product_name": "Whisper"
    },
    # META DOCS
    {
        "_id": "meta_doc1",
        "content": "Meta's Quest 3 is a mixed reality headset that offers a powerful VR and AR experience. Equipped with full-color passthrough, spatial audio, and Snapdragon XR2 Gen 2, it is ideal for gaming, collaboration, and immersive learning. It supports a wide range of apps from the Meta Quest Store and integrates with hand tracking for a controller-free experience.",
        "category": "product_info",
        "product_name": "Meta Quest 3"
    },
    {
        "_id": "meta_doc2",
        "content": "Threads is Meta's microblogging social platform designed for close-knit conversations and sharing real-time updates. Integrated with Instagram, Threads allows users to post text, images, and videos while following their interests. Its minimal interface and privacy-first design make it ideal for creators and casual users alike.",
        "category": "product_info",
        "product_name": "Threads"
    },
    {
        "_id": "meta_doc3",
        "content": "Llama 3 is Meta's latest open-source large language model (LLM), built for research and enterprise applications. It provides state-of-the-art performance on reasoning, coding, and multilingual benchmarks, and can be fine-tuned for various tasks such as summarization, chatbots, and content generation.",
        "category": "product_info",
        "product_name": "Llama 3"
    },
    # AMAZON DOCS
    {
        "_id": "amazon_doc1",
        "content": "Amazon Web Services (AWS) is the world's most comprehensive cloud computing platform, offering over 200 services including compute, storage, networking, machine learning, and security. Businesses of all sizes rely on AWS for scalable infrastructure, rapid deployment, and global reach.",
        "category": "product_info",
        "product_name": "AWS"
    },
    {
        "_id": "amazon_doc2",
        "content": "Alexa is Amazon's intelligent voice assistant that powers Echo devices and other smart home integrations. With Alexa, users can control smart devices, play music, get weather updates, and access a wide range of skills. Alexa's AI continues to evolve, providing more natural conversations and improved contextual understanding.",
        "category": "product_info",
        "product_name": "Alexa"
    },
    {
        "_id": "amazon_doc3",
        "content": "Amazon Prime is a premium membership that offers exclusive benefits like free one-day delivery, Prime Video streaming, Prime Music, and early access to deals. Prime is designed to provide convenience, entertainment, and savings for frequent Amazon shoppers.",
        "category": "product_info",
        "product_name": "Amazon Prime"
    },
    # PINECONE DOCS
    {
        "_id": "pinecone_doc1",
        "content": "Pinecone is a vector database service that enables high-speed, scalable similarity search for machine learning applications. It allows developers to store, update, and query vector embeddings in real time, making it ideal for applications like recommendation engines, semantic search, and fraud detection.",
        "category": "product_info",
        "product_name": "Pinecone Vector DB"
    },
    {
        "_id": "pinecone_doc2",
        "content": "Pinecone's hybrid search feature enables combined filtering and vector similarity search, allowing users to retrieve contextually relevant results while applying metadata constraints. This is particularly useful for e-commerce, knowledge retrieval, and personalized search solutions.",
        "category": "product_info",
        "product_name": "Hybrid Search"
    },
    {
        "_id": "pinecone_doc3",
        "content": "The Pinecone Serverless architecture eliminates the need to manage infrastructure, automatically scaling based on traffic and usage. With zero maintenance and built-in security, it simplifies deployment for AI-powered applications with minimal operational overhead.",
        "category": "product_info",
        "product_name": "Pinecone Serverless"
    }
]

def main():
    """Main function to run the chatbot"""
    try:
        config = load_config()
        chatbot = QAChatbot(config)
        
        if chatbot.initialize(business_documents):
            chatbot.run_interactive()
        else:
            print("Failed to initialize chatbot. Please check your configuration.")
            
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 