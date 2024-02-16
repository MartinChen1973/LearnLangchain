from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("tell me a joke about {foo}")
model = ChatOpenAI()
chain = prompt | model

## Start the chatbot and get the response
print("## Start the chatbot and get the response")
print(chain.invoke({"foo": "bears"}))

## Call the OpenAI API with a stop character 
print("## Attaching Stop Sequences (Bind the runtime arguments)")
chain = prompt | model.bind(stop="\n")
chain.invoke({"foo": "bears"})
print(chain.invoke({"foo": "bears"}))

## Attaching functions information
print("## Attaching functions information")
functions = [
    {
        "name": "joke",
        "description": "A joke",
        "parameters": {
            "type": "object",
            "properties": {
                "setup": {"type": "string", "description": "The setup for the joke"},
                "punchline": {
                    "type": "string",
                    "description": "The punchline for the joke",
                },
            },
            "required": ["setup", "punchline"],
        },
    }
]
chain = prompt | model.bind(function_call={"name": "joke"}, functions=functions)
print(chain.invoke({"foo": "bears"}))
