from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

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

# create a history aware retriever 创建一个历史感知的检索器 
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])
history_aware_retriever = create_history_aware_retriever(llm, retriever, prompt)

# create a document retrieval chain 创建文档检索链
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
document_chain = create_stuff_documents_chain(llm, prompt)

# connect the two chains 连接两个链
retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)

chat_history = [
    HumanMessage(content="Do you know anything about LangSmith?"), 
    AIMessage(content="Sure.")]

# Start the chatbot and get the response   启动聊天机器人并获得回复
question1 = "Can langsmith help with testing?(yes or no)";
print("User>>" + question1)
response = retrieval_chain.invoke({"input": question1, "chat_history": chat_history})
answer1 = response["answer"]
print("AI  >>" + answer1)

chat_history.append(HumanMessage(content=question1))
chat_history.append(AIMessage(content=answer1))

question2 = "Tell me how";
print("User>>" + question2)
response = retrieval_chain.invoke({"input": question2, "chat_history": chat_history})
answer2 = response["answer"]
print("AI  >>" + answer2)