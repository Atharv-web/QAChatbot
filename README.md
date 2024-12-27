# RAGify

## Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline using LangChain, FAISS for vector storage, and Ollama models for language tasks. The system supports document retrieval, question answering, web search integration, and relevance grading. It uses advanced prompt engineering techniques and includes modules for routing queries and ensuring high-quality, grounded answers.

## Features
- **Document Retrieval**: Load and split documents into chunks using LangChain's RecursiveCharacterTextSplitter.
- **Embedding**: Generate embeddings using the Ollama embeddings model and store them in a FAISS vectorstore.
- **Question Routing**: Automatically decide whether to query the vectorstore or perform a web search.
- **Answer Generation**: Generate concise and contextually accurate answers using RAG.
- **Relevance Grading**: Evaluate retrieved documents and answers for relevance and factual grounding.
- **Web Search Integration**: Use TavilySearchResults for web-based queries.

## Project Structure

### Main Components
1. **Document Loading and Processing**:
   - Load documents from URLs using WebBaseLoader.
   - Split text into manageable chunks for efficient retrieval.

2. **Vector Storage and Retrieval**:
   - Use FAISS for vector indexing and storage.
   - Retrieve top-k relevant documents for answering queries.

3. **Question Routing**:
   - A router determines whether to use the vectorstore or perform a web search based on the query.

4. **Answer Generation**:
   - Leverages the RAG technique to generate accurate answers using retrieved context.

5. **Grading and Validation**:
   - Implements graders for retrieved documents, generated answers, and hallucination detection.

### Modules
- `format_docs`: Formats retrieved documents for easier processing.
- `load_docs`: Loads documents from specified URLs.
- `split_docs`: Splits loaded documents into chunks.
- `embedder`: Embeds document chunks and saves them to a local FAISS vectorstore.
- `retrieve`: Retrieves relevant documents from the vectorstore.
- `generate`: Generates answers using the RAG technique.
- `grade_documents`: Filters out irrelevant retrieved documents.
- `web_search`: Queries the web for additional context.
- `route_question`: Routes the query to vectorstore or web search.
- `decide_generation`: Decides whether to generate an answer or search for additional context.
- `grade_generation`: Validates the generated answer for accuracy and grounding.

## Installation

### Prerequisites
- Python 3.9 or higher
- FAISS
- LangChain
- Ollama models
- TavilySearchResults
- dotenv

## Usage
1. Modify the `urls` variable to include the URLs of documents you want to load.
2. Run the pipeline to:
   - Retrieve and split documents.
   - Generate answers to user queries.
   - Grade and validate answers for relevance and grounding.

## Example Query
To test the system, you can query it with questions like:
- "What is prompt engineering?"
- "What is chain-of-thought prompting?"

## Customization
- **Embedding Model**: Modify the `OllamaEmbeddings` initialization to use a different embedding model.
- **VectorStore Parameters**: Adjust `chunk_size` and `chunk_overlap` in `RecursiveCharacterTextSplitter`.
- **Routing Logic**: Update `router_instruction` to customize the routing criteria.

## Acknowledgements
This pipeline leverages:
- [LangChain](https://www.langchain.com/) for LLM orchestration.
- [FAISS](https://github.com/facebookresearch/faiss) for efficient vector storage.
- [Ollama](https://www.ollama.ai/) for embedding and language generation.
- [Tavily](https://www.tavily.ai/) for web search integration.
