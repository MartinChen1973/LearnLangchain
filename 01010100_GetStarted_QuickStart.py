from langchain_openai import ChatOpenAI

# Load the API key from the .env file   从.env文件中加载API密钥
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# Create the OpenAI chatbot 创建聊天机器人
llm = ChatOpenAI()

# Start the chatbot and get the response   启动聊天机器人并获得回复
response = llm.invoke("how can langsmith help with testing?")

# Print the response    打印回复
print(response.content)