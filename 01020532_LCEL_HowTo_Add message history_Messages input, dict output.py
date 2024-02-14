# Load the API key from the .env file
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableParallel
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_openai import ChatOpenAI
model = ChatOpenAI()
# from langchain_community.chat_models import ChatAnthropic
# moel = ChatAnthropic(model="claude-2")

REDIS_URL = "redis://localhost:6379/0"
chain = RunnableParallel({"output_message": model})
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: RedisChatMessageHistory(session_id, url=REDIS_URL),
    output_messages_key="output_message",
)

print(chain_with_history.get_graph().print_ascii())

response = chain_with_history.invoke(
    [HumanMessage(content="What did Simone de Beauvoir believe about free will")],
    config={"configurable": {"session_id": "baz"}})
print(response)

response = chain_with_history.invoke(
    [HumanMessage(content="How did this compare to Sartre")],
    config={"configurable": {"session_id": "baz"}},
)
print(response)