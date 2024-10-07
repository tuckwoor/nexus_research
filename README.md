# Nexus Research - Chat-based AI Research Assistant

## Overview

This project is a research assistant application built using `Streamlit`, `LangChain`, and a variety of AI and vector storage tools. The application allows users to ask research-based questions and leverages different AI models (like OpenAI’s GPT and others) to provide detailed, well-informed responses while maintaining context from previous conversations.

## Features

- **Multi-model Integration**: Supports multiple AI models such as GPT-3.5, GROQ, and custom models (like LM Studio).
- **Persistent Conversation Storage**: Stores conversation history and associated context for continuity across sessions.
- **Vector-based Memory**: Uses vector embeddings and `Chroma` for efficient retrieval of past conversations.
- **Dynamic Context Handling**: Automatically integrates location and personal information into responses.
- **Web Scraping & Search Integration**: Can fetch real-time information from web searches using SearXNG and crawl websites for information.
- **Multi-agent System**: Uses a team of crewai agents to perform research and generate high-quality responses based on both live and stored data.
- **User Prompt Optimization**: Optimizes the user's prompt for the LLM to improve the quality of the response.

## Requirements

- Python 3.8+
- `Streamlit`
- `LangChain`
- `OpenAI API`
- `Chroma`
- `.env` file for API keys
- `crewai` library
- SearXNG instance

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/tuckwoor/nexus_research.git
    cd nexus_research
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up the environment variables in a `.env` file:
    ```bash
    OPENAI_API_KEY=<your-openai-api-key>
    SERPER_API_KEY=<your-serper-api-key>
    GROQ_API_KEY=<your-groq-api-key>
    SEARXNG_ENDPOINT=<your-searxng-endpoint>
    ```

4. (Optional) Set up a local SearXNG instance for web search results.

5. Remove the _sample from the .json filenames and update the contents if required.

## How to Use

1. **Run the application:**
    ```bash
    streamlit run app.py
    ```

2. **Ask a question**: Enter a research question in the input field.

3. **Conversation context**: The application will display responses based on the current conversation context. It will automatically update and store your interactions.

4. **Web Search**: If necessary, the application performs a web search to find relevant, real-time data.

5. **Conversation Memory**: Previous conversations and contexts are stored and can be accessed in future sessions.

## Key Components

- **API Models**: The script integrates various models from OpenAI, GROQ, and custom LLMs for different use cases.
- **Vector Store**: Persistent memory is handled using `Chroma`, which stores vector embeddings of conversations.
- **Context and Memory**: The assistant uses `ConversationBufferMemory` to remember previous chats and provide contextually relevant responses.
- **Web Scraping & Searching**: Web search results are fetched using SearXNG, and web pages can be scraped using BeautifulSoup for additional information.

## Next steps

- Refactoring code into modular structure
- Parallel processing and asynchronous operations to improve speed
- RAG system using RAG performance improvement techniques e.g. graphRAG
- Auto transcript of monitored youtube playlists to be incorporated into RAG system
- Response rating (thumbs up / down)
- More search engine options other than SearXNG

## Customization

You can add new models by updating the `model_configs` dictionary. Define the model class, parameters, and API details as needed.

```python
model_configs = {
    "gpt_3.5": {...},
    "groq": {...},
    "custom_model": {
        "class": CustomModelClass,
        "params": {
            "base_url": "http://localhost:1234",
            "api_key": "not-needed"
        }
    }
}
