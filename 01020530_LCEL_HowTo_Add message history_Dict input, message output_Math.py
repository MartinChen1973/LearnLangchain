# Load the API key from the .env file
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# Create a normal chain
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You're an assistant who's good at {ability}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

from langchain_openai import ChatOpenAI
chain = prompt | ChatOpenAI()
# from langchain_community.chat_models import ChatAnthropic
# chain = prompt | ChatAnthropic(model="claude-2")

# Connect to Redis. 
# To install Redis in minutes, you can follow the instructions at the end of this file.
REDIS_URL = "redis://localhost:6379/0"
session_id = "math1"

# If you want to clear the session data, you can use the following code: 
# import redis
# redis_client = redis.from_url(REDIS_URL)
# redis_client.delete("session:" + session_id)

# create a chain with history from the normal chain
history = lambda session_id: RedisChatMessageHistory(session_id, url=REDIS_URL)
chain_with_history = RunnableWithMessageHistory(
    chain,
    history,
    input_messages_key="question",
    history_messages_key="history",
)

print(chain_with_history.get_graph().print_ascii())

# invoke the chain
response = chain_with_history.invoke(
    {"ability": "math", "question": "What does cosine mean?"},
    config={"configurable": {"session_id": session_id}},
)
print(response.content)

response = chain_with_history.invoke(
    {"ability": "math", "question": "What's its inverse?"},
    config={"configurable": {"session_id": session_id}},
)
print(response.content)

# To install Redis, you can use the following code:

# in linux, run the following commands or follow the instructions at https://hub.docker.com/_/redis/
# docker pull redis
# docker run --name some-redis -d redis

# in windows, run the following commands in powershell or follow the instructions at https://hub.docker.com/r/jcreach/redis
# docker pull jcreach/redis
# docker run --name my-redis -p 6379:6379 -d jcreach/redis:5.0.14.1-lts-nanoserver-1809

# after all, you can use REDIS_URL = "redis://localhost:6379/0" to access the server.