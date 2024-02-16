from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("tell me a joke about {foo}")
model = ChatOpenAI()
chain = prompt | model

## Call with a RunnableParrallel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

map_ = RunnableParallel(foo=RunnablePassthrough())
chain = (
    map_
    | prompt | model | StrOutputParser()
)

print(chain.invoke("bears"))

## Call with a Dict
chain = (
    {"foo": RunnablePassthrough()}
    | prompt | model | StrOutputParser()
)

print(chain)

print(chain.invoke("bears"))