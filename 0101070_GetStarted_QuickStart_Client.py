# You need to run the server first. Then, run this client.
from langserve import RemoteRunnable

remote_chain = RemoteRunnable("http://localhost:8234/agent/")
response = remote_chain.invoke({
    "input": "how can langsmith help with testing?", 
    "chat_history": []})
print(response["output"])