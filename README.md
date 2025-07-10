# Optimized Business QA Chatbot

This is an optimized version of the original Jupyter notebook-based QA chatbot with significant improvements in performance, maintainability, and reliability.

## Key Optimizations

### 1. **Architecture Improvements**
- **Object-Oriented Design**: Separated concerns into distinct classes (`DocumentManager`, `LLMClient`, `PromptBuilder`, `QAChatbot`)
- **Configuration Management**: Centralized configuration using dataclasses
- **Error Handling**: Comprehensive try-catch blocks throughout the application
- **Logging**: Proper logging for debugging and monitoring

### 2. **Performance Optimizations**
- **Dynamic Host Resolution**: Automatically retrieves Pinecone index host instead of hardcoding
- **Duplicate Document Prevention**: Checks for existing documents before uploading
- **Bounded Context Cache**: Uses `deque` with maxlen to prevent memory bloat
- **Retry Logic**: Implements retry mechanism for LLM API calls
- **Efficient Data Structures**: Uses appropriate data structures for different use cases

### 3. **Memory Management**
- **Limited Context Cache**: Configurable maximum context length (default: 10)
- **Bounded Conversation History**: Configurable maximum history length (default: 6)
- **Automatic Cleanup**: Old context and history automatically removed

### 4. **Security & Configuration**
- **Environment Variable Validation**: Checks for required API keys
- **Configuration Class**: Type-safe configuration management
- **No Hardcoded Values**: All configuration externalized

### 5. **User Experience**
- **Better Error Messages**: User-friendly error handling
- **Additional Commands**: `clear` command to reset conversation history
- **Improved Prompts**: More structured and informative prompts
- **Graceful Interrupts**: Handles Ctrl+C gracefully

## Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**:
   Create a `.env` file with:
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   GEMINI_API_KEY=your_gemini_api_key
   ```

3. **Run the Chatbot**:
   ```bash
   python optimized_qa_chatbot.py
   ```

## Configuration Options

You can modify the `Config` class to adjust:
- `max_context_length`: Maximum number of documents in context cache
- `max_history_length`: Maximum conversation history length
- `top_k_results`: Number of documents to retrieve per query
- `region` and `cloud`: Pinecone deployment settings

## Comparison with Original

| Aspect | Original | Optimized |
|--------|----------|-----------|
| **Code Structure** | Monolithic functions | Object-oriented classes |
| **Error Handling** | Minimal | Comprehensive |
| **Memory Usage** | Unbounded growth | Bounded with limits |
| **Configuration** | Global variables | Centralized config class |
| **Database Operations** | Hardcoded host | Dynamic host resolution |
| **Document Upload** | Always uploads | Checks for duplicates |
| **API Calls** | No retry logic | Retry with backoff |
| **Logging** | None | Structured logging |
| **Type Safety** | No type hints | Full type annotations |

## Performance Benefits

- **Reduced API Calls**: Duplicate document prevention
- **Lower Memory Usage**: Bounded caches prevent memory leaks
- **Faster Startup**: Dynamic host resolution
- **Better Reliability**: Retry logic and error handling
- **Improved Maintainability**: Modular design and logging

## Future Enhancements

1. **Caching Layer**: Redis for persistent caching
2. **Rate Limiting**: API rate limiting protection
3. **Metrics Collection**: Performance monitoring
4. **Async Support**: Non-blocking operations
5. **Web Interface**: Flask/FastAPI web server
6. **Database Integration**: SQL database for conversation storage 