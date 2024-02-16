# Load the API key from the .env file   从.env文件中加载API密钥
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI

prompt = PromptTemplate.from_template(
    "Tell me a joke about {topic}"
).configurable_alternatives(
    ConfigurableField(id="prompt"),
    default_key="joke",
    poem=PromptTemplate.from_template("Write a short poem about {topic}. Only 4 lines."),
    chineseSaying=PromptTemplate.from_template("Write a Chinese saying about {topic}."),
)
llm = ChatOpenAI(temperature=0).configurable_alternatives(
    ConfigurableField(id="llm"),
    default_key="openai 3.5",
    gpt4=ChatOpenAI(model="gpt-4"),
)

chain = prompt | llm

# We can configure it write a poem with OpenAI Gpt4
print(chain.with_config(configurable={"prompt": "poem", "llm": "gpt4"}).invoke(
    {"topic": "bears"}
))

# Save config of Gpt4 + poem
gpt4_poem = chain.with_config(configurable={"prompt": "poem", "llm": "gpt4"})
print(gpt4_poem.invoke({"topic": "bears"}))