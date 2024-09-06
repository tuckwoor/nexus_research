import os
import json
import requests
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import time
import chromadb
import uuid
from datetime import datetime
import pytz
import geocoder
from requests.exceptions import RequestException
import streamlit as st

# Load environment variables
load_dotenv()

# Set up API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SEARXNG_ENDPOINT = os.getenv("SEARXNG_ENDPOINT")

# Define model configurations
model_configs = {
    "gpt_3.5": {
        "class": ChatOpenAI,
        "params": {
            "model": "gpt-3.5-turbo",
            "api_key": OPENAI_API_KEY,
            "temperature": 0.7
        }
    },
    "groq": {
        "class": ChatOpenAI,
        "params": {
            "base_url": "https://api.groq.com/openai/v1",
            "api_key": GROQ_API_KEY,
            "model_name": "llama-3.1-8b-instant" # llama-3.1-8b-instant / mixtral-8x7b-32768
        }
    },
    "lm_studio": {
        "class": ChatOpenAI,
        "params": {
            "base_url": "http://localhost:1234/v1",
            "api_key": "not-needed",
            "temperature": 0.7
        }
    }
}

def initialize_model(model_name):
    config = model_configs[model_name]
    return config["class"](**config["params"])

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Define paths for persistent storage
PERSIST_DIRECTORY = os.path.join(os.path.dirname(__file__), "chroma_db")
CONVERSATIONS_FILE = os.path.join(os.path.dirname(__file__), "conversations.json")
SUPER_CONTEXT_FILE = os.path.join(os.path.dirname(__file__), "super_context.json")
SECOND_LEVEL_CONTEXTS_FILE = os.path.join(os.path.dirname(__file__), "second_level_contexts.json")

# Initialize Chroma client with a persistent directory
chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

def initialize_vector_store(conversation_id):
    collection = chroma_client.get_or_create_collection(f"conversation_{conversation_id}")
    vector_store = Chroma(
        client=chroma_client,
        collection_name=f"conversation_{conversation_id}",
        embedding_function=embeddings
    )
    return vector_store

def load_conversations():
    if os.path.exists(CONVERSATIONS_FILE):
        try:
            with open(CONVERSATIONS_FILE, 'r') as f:
                conversations = json.load(f)
                print("Loaded conversations from file:", conversations)  # Debug print
                for conv_id, conv_data in conversations.items():
                    conv_data["vector_store"] = initialize_vector_store(conv_id)
                    conv_data["memory"] = ConversationBufferMemory(return_messages=True)
                    if "second_level_context" not in conv_data:
                        conv_data["second_level_context"] = "general_use"
                return conversations
        except json.JSONDecodeError as e:
            print(f"Error reading {CONVERSATIONS_FILE}. File may be corrupted.")
            print(f"Error details: {str(e)}")
            return {}
    return {}

def save_conversations(conversations):
    serializable_conversations = {}
    for conv_id, conv_data in conversations.items():
        serializable_conversations[conv_id] = {
            "title": conv_data.get("title", "Untitled"),
            "messages": conv_data.get("messages", []),
            "second_level_context": conv_data.get("second_level_context", "general_use")
        }
    
    try:
        with open(CONVERSATIONS_FILE, 'w') as f:
            json.dump(serializable_conversations, f, indent=2)
    except Exception as e:
        print(f"Error saving conversations: {str(e)}")

def load_super_context():
    if os.path.exists(SUPER_CONTEXT_FILE):
        with open(SUPER_CONTEXT_FILE, 'r') as f:
            return json.load(f)
    return {
        'personal_info': {
            'name': '',
            'occupation': '',
            'interests': []
        },
        'location': '',
        'additional_info': ''
    }

def save_super_context(super_context):
    with open(SUPER_CONTEXT_FILE, 'w') as f:
        json.dump(super_context, f, indent=2)

def update_super_context():
    super_context = load_super_context()
    uk_time = datetime.now(pytz.timezone('Europe/London'))
    super_context['date_time'] = uk_time.strftime("%Y-%m-%d %H:%M:%S %Z")
    super_context['location'] = "United Kingdom"
    g = geocoder.ip('me')
    if g.ok:
        super_context['user_city'] = g.city
        super_context['user_country'] = g.country
    
    save_super_context(super_context)
    return super_context

def get_second_level_contexts():
    if os.path.exists(SECOND_LEVEL_CONTEXTS_FILE):
        with open(SECOND_LEVEL_CONTEXTS_FILE, 'r') as f:
            return json.load(f)
    else:
        print(f"Warning: {SECOND_LEVEL_CONTEXTS_FILE} not found. Using default contexts.")
        return {
            "general_use": "You are a general-purpose assistant."
        }

def create_new_conversation(second_level_context="general_use"):
    conversation_id = str(uuid.uuid4())
    st.session_state.conversations[conversation_id] = {
        "title": "New Conversation",
        "messages": [],
        "vector_store": initialize_vector_store(conversation_id),
        "memory": ConversationBufferMemory(return_messages=True),
        "second_level_context": second_level_context
    }
    save_conversations(st.session_state.conversations)
    return conversation_id

def get_conversation_summary(messages):
    if messages:
        return messages[0]['content'][:50] + "..."
    return None

def crawl_webpage(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract text content from the main body
        main_content = soup.find('body')
        if main_content:
            # Remove script and style elements
            for script in main_content(["script", "style"]):
                script.decompose()
            text = ' '.join(main_content.stripped_strings)
            # Limit the text to around 1000 words
            words = text.split()[:1000]
            return ' '.join(words)
        return ""
    except Exception as e:
        print(f"Error crawling {url}: {str(e)}")
        return ""

def searxng_search(query_input):
    url = SEARXNG_ENDPOINT
    if not url:
        raise ValueError("SEARXNG_ENDPOINT is not set in the .env file")
    
    # Handle different input types
    if isinstance(query_input, dict):
        query = query_input.get('query', '') or query_input.get('tool_input', {}).get('string', '')
    else:
        query = str(query_input)
    
    if not query:
        print("Error: Empty query received")
        return []

    params = {
        "q": query,
        "format": "json",
        "engines": "qwant,duckduckgo,wikipedia",
        "language": "en",
        "time_range": "month"
    }
    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            print(f"Performing search with query: {query}")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            results = response.json()
            
            formatted_results = []
            for result in results.get('results', [])[:5]:
                if any(keyword in result.get('title', '').lower() or keyword in result.get('content', '').lower() 
                       for keyword in query.lower().split()):
                    formatted_results.append({
                        'title': result.get('title', ''),
                        'snippet': result.get('content', ''),
                        'link': result.get('url', '')
                    })
            
            if not formatted_results:
                print(f"No relevant results found for query: {query}")
            else:
                print(f"Found {len(formatted_results)} relevant results for query: {query}")
            
            return formatted_results
        except RequestException as e:
            print(f"Error performing SearXNG search (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Unable to perform search.")
                return []

### NEW FUNCTIONS FOR QUERY ANALYSIS, OPTIMIZATION, AND EXPANSION ###

def analyze_query(query, chat_history, super_context):
    # List of keywords that always trigger a search
    search_keywords = ['latest', 'current', 'recent', 'last', 'new', 'update', 'today', 'yesterday', 'this week', 'this month']
    
    # Check if any of the search keywords are in the query
    if any(keyword in query.lower() for keyword in search_keywords):
        return True

    # Check for specific patterns that might indicate a need for current information
    time_related_patterns = [
        r'\d+\s*(day|week|month|year)s?\s*(ago|old)',  # e.g., "3 days ago", "2 weeks old"
        r'(this|last|next)\s+(day|week|month|year)',   # e.g., "this week", "last month"
        r'\d{4}',  # Any four-digit year
        r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}',  # Date patterns like "Aug 15"
    ]
    if any(re.search(pattern, query.lower()) for pattern in time_related_patterns):
        return True

    # If the answer is already in the chat history or super context, skip the search
    for message in chat_history:
        if query.lower() in message['content'].lower():
            return False
    
    for _, value in super_context.items():
        if isinstance(value, str) and query.lower() in value.lower():
            return False
        elif isinstance(value, dict):
            for sub_value in value.values():
                if isinstance(sub_value, str) and query.lower() in sub_value.lower():
                    return False
                elif isinstance(sub_value, list):
                    for item in sub_value:
                        if isinstance(item, str) and query.lower() in item.lower():
                            return False

    # If we've made it this far, we should probably search
    return True

def optimize_query(query, chat_history, super_context):
    # Use known context to optimize the query
    context_keywords = []

    # Extract keywords from the super_context
    if 'personal_info' in super_context:
        context_keywords += super_context['personal_info'].get('interests', [])

    # Use relevant chat history
    for message in chat_history:
        if hasattr(message, 'type') and message.type == 'human':
            context_keywords += re.findall(r'\b\w+\b', message.content.lower())
        elif isinstance(message, dict) and message.get('role') == 'human':
            context_keywords += re.findall(r'\b\w+\b', message['content'].lower())

    # Prioritize query keywords over context keywords
    query_keywords = re.findall(r'\b\w+\b', query.lower())
    
    # Combine user query with context and system knowledge, prioritizing query keywords
    optimized_query = ' '.join(query_keywords + [kw for kw in context_keywords if kw not in query_keywords])
    
    return optimized_query

def expand_query(query):
    # Expand the query by adding synonyms or related terms, removing ambiguity, or specifying info type
    synonyms = {
        'latest': ['recent', 'current'],
        'find': ['locate', 'search for'],
        'define': ['meaning', 'definition'],
        'examples': ['sample', 'instance'],
    }
    
    expanded_query = query
    for word, expansions in synonyms.items():
        if word in query.lower():
            expanded_query += " " + " ".join(expansions)
    
    return expanded_query

# Define a list of stop words
stop_words = set(['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it',
                  'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with'])

def search_with_print(query_input, chat_history, super_context):
    if isinstance(query_input, dict):
        query = query_input.get('query', '')
    else:
        query = str(query_input)

    print(f"Performing search with query: {query}")
    search_results = searxng_search(query)

    if not search_results:
        print("No search results found. Trying alternative search...")
        # Try an alternative search by splitting the query
        split_query = ' '.join(query.split()[:3])  # Use first 3 words
        print(f"Performing alternative search with query: {split_query}")
        search_results = searxng_search(split_query)

    if not search_results:
        return "No relevant information found from web search. Please try rephrasing your query or asking a more specific question."
    
    return search_results

# Update the search_tool definition
search_tool = Tool(
    name="SearXNGSearch",
    func=lambda query_input: searxng_search(query_input),
    description="A tool for performing web searches using a local SearXNG server to get relevant results. If no results are found, try rephrasing the query or breaking it into smaller parts."
)

def analyze_initial_results(results):
    # Extract keywords from the search results
    keywords = []
    for result in results:
        keywords.extend(re.findall(r'\b\w+\b', result.get('title', '') + ' ' + result.get('snippet', '')))
    
    # Count the most common keywords
    keyword_counts = Counter(keywords)
    
    # Get the top 5 most common keywords, excluding stop words
    top_keywords = [word for word, _ in keyword_counts.most_common(10) if word.lower() not in stop_words][:5]
    
    return ' '.join(top_keywords)

def needs_further_search(results, query):
    # Implement logic to determine if further search is needed
    # For example, if the initial results don't seem to fully answer the query
    return len(results) < 3  # Simple example: if we have fewer than 3 results, do another search

def update_vector_store(text):
    st.session_state.vector_store.add_texts([text])

def get_relevant_context(query, k=5):
    current_conv = st.session_state.conversations[st.session_state.current_conversation]
    relevant_docs = current_conv["vector_store"].similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in relevant_docs])

# Add this new function
def get_recent_conversation_context(chat_history, n=3):
    recent_context = []
    for message in reversed(chat_history[-6:]):  # Look at the last 6 messages
        if len(recent_context) >= n * 2:  # We want n pairs of human-AI interactions
            break
        if isinstance(message, (HumanMessage, AIMessage)):
            recent_context.insert(0, {
                "role": "human" if isinstance(message, HumanMessage) else "ai",
                "content": message.content
            })
        elif isinstance(message, dict):
            recent_context.insert(0, message)
    return recent_context

# Add this new class for JSON serialization
class MessageEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (HumanMessage, AIMessage)):
            return {
                "type": obj.__class__.__name__,
                "content": obj.content
            }
        return super().default(obj)

def generate_task_description(user_input, conversation_history, super_context, second_level_context, initial_search_summary):
    # Get the full context description
    second_level_contexts = get_second_level_contexts()
    full_second_level_context = second_level_contexts.get(second_level_context, "No specific context provided.")

    # Get recent conversation context
    recent_context = get_recent_conversation_context(conversation_history)
    
    # Get relevant context from vector store
    relevant_context = get_relevant_context(user_input)

    task_description = f"""
    IMPORTANT: Follow these steps in order when answering questions:
    1. User Input (MOST IMPORTANT): {user_input}
       Always prioritize addressing the user's specific question or request.

    2. Recent Conversation Context:
    {json.dumps(recent_context, indent=2, cls=MessageEncoder)}
    
    3. Relevant Context from Previous Conversations:
    {relevant_context}

    4. Initial Search Summary:
    {initial_search_summary}
    
    5. Second-level Context (ONLY IF RELEVANT):
    {full_second_level_context}
    
    6. Super Context (ONLY IF RELEVANT):
    {json.dumps(super_context, indent=2)}

    Based on the above information, please provide a comprehensive response to the user's input, ensuring that the response directly addresses the user's question or request. Maintain continuity with the recent conversation context and use the relevant context from previous conversations to inform your answer.

    You have access to a search tool. Use it to gather additional information as needed. Don't limit yourself to the initial search results - perform multiple searches if necessary to gather comprehensive and up-to-date information.
    """

    return {
        'research_task_description': f"Research the following topic, focusing on the user's specific input and maintaining continuity with the conversation context: {task_description}",
        'research_task_expected_output': "Comprehensive research findings that directly address the user's input, prioritizing relevant and current information while maintaining context from the ongoing conversation.",
        'writing_task_description': "Based on the research findings, create an engaging and informative response that directly answers the user's question or request, focusing on the most relevant and up-to-date information. Ensure the response maintains continuity with the ongoing conversation.",
        'writing_task_expected_output': "A well-written, engaging response that directly addresses the user's input, based on the research findings, emphasizing relevant and current data, without any fabricated information. The response should flow naturally from the ongoing conversation."
    }

# Add this new function
def optimize_prompt(task_description: str) -> str:
    optimization_prompt = f"""
    Given the following task description, create an optimized prompt for an LLM:

    {task_description}

    Your task:
    1. Remove unnecessary instructions and conditions.
    2. Focus directly on guiding the LLM to answer the specific question.
    3. Cut out any irrelevant details.
    4. Maintain a clear focus on the most important aspects pertinent to the question.
    5. Only include information from the original context that is directly tied to the question.
    6. Remove any reference to preferred sources of information or websites to use, unless specified in the user input or the second-level context.
    7. Avoid any reference to meta-information or preferences about how to prioritize the user's request.

    Provide the optimized prompt below:
    """

    llm = initialize_model("groq")  # Or whichever model you prefer
    optimized_prompt = llm.invoke(optimization_prompt)  # Use invoke instead of __call__
    
    # Extract the string content from the AIMessage
    optimized_prompt_str = optimized_prompt.content if hasattr(optimized_prompt, 'content') else str(optimized_prompt)
    
    # Print the optimized prompt in the terminal debug window
    print("Optimized Prompt:")
    print(optimized_prompt_str)
    
    return optimized_prompt_str

# Global variable to track current agent and task
current_agent = {"name": "Initializing...", "task": ""}

# Wrapper function for execute_task
def execute_task_wrapper(agent, task):
    global current_agent
    current_agent["name"] = agent.role
    current_agent["task"] = task.description[:50] + "..."
    return agent.execute_task(task)

# Define agents configurations with specified models
def create_agents(super_context, full_second_level_context):
    return [
        Agent(
            role='Initial Search Analyst',
            goal='Analyze initial search results and create a concise summary',
            backstory='You are a quick-thinking analyst capable of distilling large amounts of information into clear, concise summaries.',
            verbose=True,
            allow_delegation=False,
            llm=initialize_model("groq"),
            max_iterations=2,
            max_rpm=10,
            time_limit=120  # 2 minutes
        ),
        Agent(
            role='Senior Research Analyst',
            goal='Conduct comprehensive research on given topics',
            backstory='You are an experienced researcher with a keen eye for detail and a knack for finding valuable information.',
            verbose=True,
            allow_delegation=False,
            tools=[search_tool],
            llm=initialize_model("groq"),
            max_iterations=20,  # Increased from 5 to 10
            max_rpm=10,
            time_limit=600  # Increased from 300 to 600 (10 minutes)
        ),
        Agent(
            role='Content Writer',
            goal='Create engaging and informative content based on research',
            backstory='You are a skilled writer with expertise in various subjects and the ability to craft compelling narratives.',
            verbose=True,
            allow_delegation=False,
            llm=initialize_model("groq"),
            max_iterations=3,
            max_rpm=10,
            time_limit=180  # 3 minutes
        )
    ]

def save_second_level_context(name, description):
    contexts = get_second_level_contexts()
    contexts[name] = description
    with open(SECOND_LEVEL_CONTEXTS_FILE, 'w') as f:
        json.dump(contexts, f, indent=2)

def delete_second_level_context(name):
    contexts = get_second_level_contexts()
    if name in contexts:
        del contexts[name]
        with open(SECOND_LEVEL_CONTEXTS_FILE, 'w') as f:
            json.dump(contexts, f, indent=2)
        return True
    return False

def delete_conversation(conv_id):
    if conv_id in st.session_state.conversations:
        # Remove from in-memory state
        del st.session_state.conversations[conv_id]
        
        # Update JSON file
        try:
            with open(CONVERSATIONS_FILE, 'r') as f:
                conversations = json.load(f)
            
            if conv_id in conversations:
                del conversations[conv_id]
            
            with open(CONVERSATIONS_FILE, 'w') as f:
                json.dump(conversations, f, indent=2)
        except Exception as e:
            print(f"Error updating conversations file: {str(e)}")
            return False
        
        if st.session_state.current_conversation == conv_id:
            # If the deleted conversation was the current one, create a new conversation
            st.session_state.current_conversation = create_new_conversation("general_use")
            st.session_state.chat_history = []
        
        return True
    return False

def main():
    st.set_page_config(layout="wide")
    
    # Initialize session state variables
    if 'conversations' not in st.session_state:
        st.session_state.conversations = load_conversations()
    
    # Debug print
    print("Loaded conversations:", st.session_state.conversations)
    
    if 'current_conversation' not in st.session_state or st.session_state.current_conversation not in st.session_state.conversations:
        st.session_state.current_conversation = create_new_conversation()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'selected_context' not in st.session_state:
        st.session_state.selected_context = "general_use"

    # Load super context
    super_context = load_super_context()

    # Sidebar
    st.sidebar.title("NEXUS Research")  # Sticky title

    # Super Context Editor (Collapsible)
    with st.sidebar.expander("Personal Information", expanded=False):
        with st.form("super_context_form"):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.text("Name:")
                st.text("Occupation:")
                st.text("Interests:")
                st.text("Location:")
                st.text("Additional Info:")
            with col2:
                name = st.text_input("Name", value=super_context.get('personal_info', {}).get('name', ''), label_visibility="collapsed")
                occupation = st.text_input("Occupation", value=super_context.get('personal_info', {}).get('occupation', ''), label_visibility="collapsed")
                interests = st.text_area("Interests", value='\n'.join(super_context.get('personal_info', {}).get('interests', [])), label_visibility="collapsed")
                location = st.text_input("Location", value=super_context.get('location', ''), label_visibility="collapsed")
                additional_info = st.text_area("Additional Info", value=super_context.get('additional_info', ''), label_visibility="collapsed")
            
            if st.form_submit_button("Save Personal Information"):
                new_super_context = {
                    'personal_info': {
                        'name': name,
                        'occupation': occupation,
                        'interests': [interest.strip() for interest in interests.split('\n') if interest.strip()]
                    },
                    'location': location,
                    'additional_info': additional_info
                }
                save_super_context(new_super_context)
                st.success("Personal Information saved successfully!")
                super_context = new_super_context

    # Update date and time
    uk_time = datetime.now(pytz.timezone('Europe/London'))
    super_context['date_time'] = uk_time.strftime("%Y-%m-%d %H:%M:%S %Z")

    # New Conversation button
    if st.sidebar.button("New Conversation", help="Start a new conversation"):
        st.session_state.current_conversation = create_new_conversation(st.session_state.selected_context)
        st.session_state.chat_history = []

    # List of conversations
    st.sidebar.write("Previous Conversations:")
    
    def get_last_timestamp(conv_data):
        messages = conv_data.get("messages", [])
        if messages:
            return messages[-1].get("timestamp", 0)
        return 0

    sorted_conversations = sorted(
        st.session_state.conversations.items(),
        key=lambda x: get_last_timestamp(x[1]),
        reverse=True
    )
    
    # Grouping by date
    grouped_conversations = {}
    for conv_id, conv_data in sorted_conversations:
        summary = get_conversation_summary(conv_data["messages"])
        if summary:
            conv_date = datetime.fromtimestamp(get_last_timestamp(conv_data)).strftime('%Y-%m-%d')
            if conv_date not in grouped_conversations:
                grouped_conversations[conv_date] = []
            grouped_conversations[conv_date].append((conv_id, summary))

    # Debug print
    print("Number of conversations:", len(st.session_state.conversations))
    
    # Debug print
    print("Sorted conversations:", sorted_conversations)
    
    # Debug print
    print("Grouped conversations:", grouped_conversations)

    for date, conversations in grouped_conversations.items():
        with st.sidebar.expander(date, expanded=False):
            for conv_id, summary in conversations:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    if st.button(summary, key=f"select_{conv_id}"):
                        st.session_state.current_conversation = conv_id
                        st.session_state.chat_history = st.session_state.conversations[conv_id]["messages"]
                        st.session_state.selected_context = st.session_state.conversations[conv_id].get("second_level_context", "general_use")
                with col2:
                    # Empty column for spacing
                    pass
                with col3:
                    if st.button("🗑️", key=f"delete_{conv_id}", help="Delete this conversation"):
                        if delete_conversation(conv_id):
                            st.success("Conversation deleted successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to delete conversation.")

    # Second-level context editor
    with st.sidebar.expander("Edit Second-level Contexts", expanded=False):
        second_level_contexts = get_second_level_contexts()
        
        # Dropdown to select existing context or create new
        context_options = list(second_level_contexts.keys()) + ["Create New"]
        selected_context_to_edit = st.selectbox("Select Context to Edit", context_options)
        
        if selected_context_to_edit == "Create New":
            new_context_name = st.text_input("New Context Name")
            new_context_description = st.text_area("New Context Description")
        else:
            new_context_name = st.text_input("Context Name", value=selected_context_to_edit)
            new_context_description = st.text_area("Context Description", value=second_level_contexts[selected_context_to_edit])
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Context"):
                if new_context_name and new_context_description:
                    save_second_level_context(new_context_name, new_context_description)
                    st.success(f"Context '{new_context_name}' saved successfully!")
                    # Refresh the contexts
                    second_level_contexts = get_second_level_contexts()
                else:
                    st.error("Please provide both name and description for the context.")
        
        with col2:
            if selected_context_to_edit != "Create New":
                if st.button("Delete Context"):
                    if delete_second_level_context(selected_context_to_edit):
                        st.success(f"Context '{selected_context_to_edit}' deleted successfully!")
                        # Refresh the contexts
                        second_level_contexts = get_second_level_contexts()
                    else:
                        st.error("Failed to delete context.")

    # Main content area
    main_container = st.container()
    with main_container:
        # Display current task status at the top
        st.subheader(f"Current Task: {current_agent['name']} - {current_agent['task']}")

        # Display chat history in a scrollable section
        st.write("---")
        st.markdown(f"**Chat History** (scrollable)")
        for message in st.session_state.chat_history:
            st.chat_message(message["role"]).write(message["content"])

        # Dropdown for selecting conversation context
        second_level_contexts = get_second_level_contexts()
        selected_context = st.selectbox(
            "Select Conversation Context", 
            list(second_level_contexts.keys()), 
            index=list(second_level_contexts.keys()).index(st.session_state.selected_context) if st.session_state.selected_context in second_level_contexts else list(second_level_contexts.keys()).index("general_use")
        )
        st.session_state.selected_context = selected_context

    # User input at the bottom
    user_input = st.chat_input("Ask a research question:")

    if user_input:
        # Update super context with current date and time
        super_context = update_super_context()

        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "human",
            "content": user_input,
            "timestamp": time.time()
        })
        st.chat_message("human").write(user_input)

        # Perform research
        with st.spinner("Researching..."):
            try:
                current_conv = st.session_state.conversations[st.session_state.current_conversation]

                # Update vector store with user input
                current_conv["vector_store"].add_texts([user_input])

                conversation_history = current_conv["memory"].chat_memory.messages

                # Perform initial search
                initial_search_results = search_tool.run(user_input)

                # Get the full context description
                second_level_contexts = get_second_level_contexts()
                full_second_level_context = second_level_contexts.get(st.session_state.selected_context, "No specific context provided.")

                print(f"Full second-level context being used: {full_second_level_context}")

                # Create tasks and crew
                agents = create_agents(super_context, full_second_level_context)
                
                initial_analysis_task = Task(
                    description=f"Analyze the following initial search results and create a concise summary (250 words or less) that captures the key information related to the user's query: '{user_input}'. Initial search results: {initial_search_results}",
                    agent=agents[0],
                    expected_output="A concise summary (250 words or less) of the initial search results, focusing on information relevant to the user's query."
                )

                # Execute the initial analysis task
                initial_search_summary = execute_task_wrapper(agents[0], initial_analysis_task)

                task_descriptions = generate_task_description(
                    user_input,
                    conversation_history,
                    super_context,
                    st.session_state.selected_context,
                    initial_search_summary
                )

                # Optimize the research task description
                optimized_research_description = optimize_prompt(task_descriptions['research_task_description'])

                research_task = Task(
                    description=optimized_research_description,
                    agent=agents[1],
                    expected_output=task_descriptions['research_task_expected_output']
                )
                
                writing_task = Task(
                    description=task_descriptions['writing_task_description'],
                    agent=agents[2],
                    expected_output=task_descriptions['writing_task_expected_output']
                )

                crew = Crew(
                    agents=agents,
                    tasks=[research_task, writing_task],
                    verbose=True,
                    process=Process.sequential
                )

                # Run the crew
                result = crew.kickoff()
                result_str = str(result)

                if "Agent stopped due to iteration limit or time limit" in result_str:
                    result_str = "I apologize, but I couldn't complete the research within the given time limit. Here's what I found so far:\n\n" + result_str

                # Add AI response to chat history
                st.session_state.chat_history.append({
                    "role": "ai",
                    "content": result_str,
                    "timestamp": time.time()
                })
                st.chat_message("ai").write(result_str)

                # Update conversation data
                current_conv["messages"] = st.session_state.chat_history
                current_conv["memory"].chat_memory.add_user_message(user_input)
                current_conv["memory"].chat_memory.add_ai_message(result_str)
                current_conv["vector_store"].add_texts([result_str])
                current_conv["second_level_context"] = st.session_state.selected_context  # Save the selected context

            except Exception as e:
                error_message = f"An error occurred during research: {str(e)}"
                st.error(error_message)
                print(f"Error details: {e}")  # Add this line for more detailed error information
                st.session_state.chat_history.append({
                    "role": "ai",
                    "content": error_message,
                    "timestamp": time.time()
                })

        # Save updated conversations
        save_conversations(st.session_state.conversations)

if __name__ == "__main__":
    main()
