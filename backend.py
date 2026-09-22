from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages

import sqlite3
import requests
import math

from langchain_tavily import TavilySearch
from langchain_core.tools import tool


load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b")

device = "cuda" if torch.cuda.is_available() else "cpu"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)

def ingest_rag_document(file_path):
    DB_PATH = "faiss_db"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(DB_PATH)

def get_retriever():
    DB_PATH = "faiss_db"
    vector_store = FAISS.load_local(
        folder_path=DB_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vector_store.as_retriever(
        search_type='similarity', 
        search_kwargs={'k':4}
    )

    return retriever

#tools
search_tool = TavilySearch(
    max_results=5,
    topics="general",
    search_depth="advanced"
)

@tool

def rag_tool(query: str) -> str:
    """ Retrieve relevant information from the PDF information"""

    retriever = get_retriever()
    documents = retriever.invoke(query)

    if not documents:
        return "No relevent information is found in the PDF."

    formatted_documents = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "Unknown page")

        formatted_documents.append(
            f"Document {index}\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content: {document.page_content}"
        )

    return "\n\n".join(formatted_documents)

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        }

        result = eval(expression, {"__builtins__": None}, allowed)
        return str(result)

    except Exception as e:
        return f"Error evaluating expression: {e}"

@tool
def get_stock_price(ticker: str) -> str:
    """Fetch the current stock price for a given ticker symbol."""
    
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey=EN8XD2O6RL68KTJ2"
    response = requests.get(url)
    return str(response.json())

tools = [search_tool, calculator, get_stock_price, rag_tool]

llm_with_tools = llm.bind_tools(tools)

# State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):

    system_message = system_message = SystemMessage(
    content=
        "You are a helpful Agentic Chatbot with access to several tools.\n\n"
        
        "Tool usage instructions:\n"
        "- Use 'rag_tool' for questions about the uploaded PDF or document.\n"
        "- Always retrieve relevant document content before answering PDF-related questions.\n"
        "- Use 'search_tool' for current events, recent information or information\n"
        "that requires an internet search.\n"
        "- Use 'calculator' for mathematical calculations. Do not calculate complex\n"
        "expressions manually when the calculator is available.\n"
        "- Use 'get_stock_price' when the user asks for the current price of a stock.\n"
        # "- Use 'get_current_weather' when the user asks about current weather for a location.\n\n"
        
        "Answer general questions directly when no tool is required.\n"
        "Do not invent information from the uploaded document.\n"
        "If the user asks about a PDF but no document is available, ask them to upload a PDF.\n"
        "After receiving a tool result, provide a clear and helpful final answer."
)

    messages = [system_message] + state["messages"]
    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}

tool_node = ToolNode(tools)

#checkpointer
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpoint = SqliteSaver(conn)

graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpoint)

#Helpper function for streamlit frontend
def get_all_threads():
    threads_ids = checkpoint.list(None)
    all_threads = set()
    for thread_id in threads_ids:
        all_threads.add(thread_id.config['configurable']['thread_id'])

    return list(all_threads)


# config = {"configurable": {"thread_id": "default_thread-1"}}
