from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load the API key from the .env file 从.env文件中加载API密钥
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# Create the OpenAI chatbot 创建聊天机器人
llm = ChatOpenAI()

# create a prompt 创建提示词
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
])

# Create an output parser 创建输出解析器
output_parser = StrOutputParser()

# Create and invoke a chain 创建并调用链
chain = prompt | llm | output_parser

result = chain.invoke({"input": "how can langsmith help with testing?"})
# response = chain.invoke({"input": "how can langsmith help with testing? Keep the answer short."})

# Print the response
print(result)