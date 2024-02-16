from langchain_openai import ChatOpenAI

from ... import config

# use the config to load the API key, or use the following code to load the API key from the .env file (Attention that the .env file must be in the same folder of current file ). 
# from dotenv import load_dotenv, find_dotenv
# _ = load_dotenv(find_dotenv())

# Create the OpenAI chatbot
llm = ChatOpenAI()

# Start the chatbot and get the response   启动聊天机器人并获得回复
response = llm.invoke("how can langsmith help with testing?")

# Print the response    打印回复
print(response.content)