# 0. Load the API key from the .env file 从.env文件中加载API密钥
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import time
from langchain_core.runnables import RunnableParallel

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
output_parser = StrOutputParser()

chain1 = ChatPromptTemplate.from_template("tell me a joke about {topic}") | model
chain2 = (
    ChatPromptTemplate.from_template("write a short (2 line) poem about {topic}")
    | model
)

# Sequential chain
start_time = time.perf_counter()
print(chain1.invoke({"topic": "bears"}))
print(chain2.invoke({"topic": "cats"}))
end_time = time.perf_counter()
print(f">> Sequential chains took {end_time - start_time:0.4f} seconds")

# Parallel chain
combined = RunnableParallel({"joke": chain1, "poem": chain2})
start_time = time.perf_counter()
print(combined.invoke({"topic": "bears"}))
end_time = time.perf_counter()
print(f">> Parallel chain took {end_time - start_time:0.4f} seconds")

# Batch
start_time = time.perf_counter()
print(combined.batch([{"topic": "bears"}, {"topic": "cats"}]))
end_time = time.perf_counter()
print(f">> Parallel chain batch took {end_time - start_time:0.4f} seconds")
