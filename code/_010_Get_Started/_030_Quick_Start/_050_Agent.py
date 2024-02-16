from langchain_openai import ChatOpenAI

from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage

# pip install langchainhub
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor

# Load the API key from the .env file 从.env文件中加载API密钥
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# Create the OpenAI chatbot 创建聊天机器人
llm = ChatOpenAI()

# Load the documentation 加载文档
loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()

# Create the vector store and retriever 创建向量存储和检索器
embeddings = OpenAIEmbeddings()

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

retriever = vector.as_retriever()

# Create first retriever tool based on the document retriever 基于文档检索器创建第一个检索器工具
from langchain.tools.retriever import create_retriever_tool
document_retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)

# Create second retriever tool based on the tavily search 基于tavily搜索创建第二个检索器工具
from langchain_community.tools.tavily_search import TavilySearchResults
tavily_search_retriever_tool = TavilySearchResults()

# Wrap the tools in a list 将工具包装在列表中
tools = [document_retriever_tool, tavily_search_retriever_tool]

# Create an agent to use the tools 创建一个智能体来使用这些工具
prompt = hub.pull("hwchase17/openai-functions-agent") # Get the prompt to use - you can modify this!
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Ask the question of the Langsmith (should be answered by document_retriever_tool) 
# 问langsmith的问题（应该由document_retriever_tool回答） 
agent_executor.invoke({"input": "how can langsmith help with testing?"})

# Ask the chat-history-based question of the Langsmith (should be answered by document_retriever_tool)
# 问Langsmith的基于对话历史的问题（应该由document_retriever_tool回答）  
chat_history = [HumanMessage(content="Can LangSmith help test my LLM software applications?"), AIMessage(content="Yes!")]
agent_executor.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})

# Ask the question of the weather (should be answered by tavily_search_retriever_tool) 
# 问天气的问题（应该由tavily_search_retriever_tool回答）
agent_executor.invoke({"input": "what is the weather in SF today?"})

