from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load the API key from the .env file 从.env文件中加载API密钥
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# Create the OpenAI chatbot 创建聊天机器人
llm = ChatOpenAI()

# Load the documentation 加载文档
loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()

# Create the vector store 创建向量存储
embeddings = OpenAIEmbeddings()

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

# for doc in documents:
#     print("--------------------")
#     print(doc)

# create a prompt 创建提示词
prompt = ChatPromptTemplate.from_template(
    """
    You are world class technical documentation writer.
    Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}""")

# Create and invoke a document chain 创建文档链 
document_chain = create_stuff_documents_chain(llm, prompt)

# Create and invoke a retrieval chain 创建并调用检索链
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
print(response["answer"])
# print("==========================")
# print(response)