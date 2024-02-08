# Load the API key from the .env file   从.env文件中加载API密钥
from dotenv import load_dotenv, find_dotenv

from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Define a model with a configurable field
model = ChatOpenAI().configurable_fields(
    max_tokens=ConfigurableField(
        id="llm_max_tokens",
        name="LLM Max Tokens",
        description="The maximum number of tokens to generate in the LLM",
    )
)

# Now max tokens is a configurable field, so we can change it at runtime
print(model.with_config(configurable={"llm_max_tokens": 50}).invoke("What's 3 * 2?"))
print(model.with_config(configurable={"llm_max_tokens": 10}).invoke("What's 3 * 2?"))
print(model.with_config(configurable={"llm_max_tokens": 5}).invoke("What's 3 * 2?"))

# Use a chain with configurable fields
prompt = PromptTemplate.from_template("Answer the following question: {question}?")
chain =  prompt | model
print(chain.with_config(configurable={"llm_max_tokens": 50}).invoke({"question": "What is 3 * 2"}))
print(chain.with_config(configurable={"llm_max_tokens": 10}).invoke({"question": "What is 3 * 2"}))
print(chain.with_config(configurable={"llm_max_tokens": 5}).invoke({"question": "What is 3 * 2"}))

# Print stream seperated by "|"
for trunk in chain.with_config(configurable={"llm_max_tokens": 10}).stream({"question": "What's 3 * 2?"}):
    print(trunk.content, end="|", flush=True)
