# Load the API key from the .env file   从.env文件中加载API密钥
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI

prompt = PromptTemplate.from_template(
    "Tell me a joke about {topic}"
).configurable_alternatives(
    # This gives this field an id
    # When configuring the end runnable, we can then use this id to configure this field
    ConfigurableField(id="prompt"),
    # This sets a default_key.
    # If we specify this key, the default LLM (ChatAnthropic initialized above) will be used
    default_key="joke",
    # This adds a new option, with name `poem`
    poem=PromptTemplate.from_template("Write a short poem about {topic}. Only 4 lines."),
    chineseSaying=PromptTemplate.from_template("Write a Chinese saying about {topic}."),
    # You can add more configuration options here
)
llm = ChatOpenAI(temperature=0)
chain = prompt | llm

print(chain.invoke({"topic": "bears"}))
print(chain.with_config(configurable={"prompt": "poem"}).invoke({"topic": "bears"}))
print(chain.with_config(configurable={"prompt": "chineseSaying"}).invoke({"topic": "bears"}))