# Load the API key from the .env file   从.env文件中加载API密钥
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI

prompt = PromptTemplate.from_template("Tell me a joke about {topic}. And at the end , sign your company name and model name with version(for example: 'OpenAI, gpt-3.5-burbo', but use your own information instead) in format: - by YourCompany, YouModelNameWithVersion")
llm = ChatOpenAI(temperature=0).configurable_alternatives(
    # This gives this field an id
    # When configuring the end runnable, we can then use this id to configure this field
    ConfigurableField(id="llm"),
    # This sets a default_key.
    # If we do not specify this key, the default LLM (ChatOpenAI 3.5 initialized above) will be used
    default_key="openai 3.5",
    # This adds a new option, with name `anthropic` that is equal to `ChatAnthropic()`
    # anthropic = ChatAnthropic(temperature=0),
    # This adds a new option, with name `gpt4` that is equal to `ChatOpenAI(model="gpt-4")`
    gpt4=ChatOpenAI(model="gpt-4"),
    # You can add more configuration options here
)
chain = prompt | llm

print(chain.invoke({"topic": "bears"}))
print(chain.with_config(configurable={"llm": "gpt4"}).invoke({"topic": "bears"}))