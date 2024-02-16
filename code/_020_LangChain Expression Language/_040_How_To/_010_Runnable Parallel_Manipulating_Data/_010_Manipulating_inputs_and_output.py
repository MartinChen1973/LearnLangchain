# Load the API key from the .env file   从.env文件中加载API密钥
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

vectorstore = FAISS.from_texts(
    ["harrison worked at kensho"], embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI()

# Use dict as input. The type conversion (to RunnableParallel) is done automatically.
input = {"context": retriever, "question": RunnablePassthrough()}
# Use RunnableParallel 1
# input = RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
# Use RunnableParallel 2
# input = RunnableParallel(context = retriever, question = RunnablePassthrough())

retrieval_chain = input | prompt | model | StrOutputParser() 
response = retrieval_chain.invoke("where did harrison work?")
print(response)