from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("tell me a joke about {foo}")
model = ChatOpenAI()

## PromptTemplate + LLM + OutputParser (PMO Chain)
print("## PromptTemplate + LLM + OutputParser")
from langchain_core.output_parsers import StrOutputParser

chain = prompt | model | StrOutputParser()

print(chain.invoke({"foo": "bears"}))

## Functions Output Parser
print("## Functions Output Parser")
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

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

chain = (
    prompt
    | model.bind(function_call={"name": "joke"}, functions=functions)
    | JsonOutputFunctionsParser()
)

print(chain.invoke({"foo": "bears"}))

## Functions Output Parser with key name
print("## Functions Output Parser with key name")
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser

chain = (
    prompt
    | model.bind(function_call={"name": "joke"}, functions=functions)
    | JsonKeyOutputFunctionsParser(key_name="setup")
)

print(chain.invoke({"foo": "bears"}))

# Attaching tools information
# Langchain is shifting from "functions" to "tools," so prioritize tools over functions.
# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "joke",
#             "description": "A joke",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "setup": {"type": "string", "description": "The setup for the joke"},
#                     "punchline": {
#                         "type": "string",
#                         "description": "The punchline for the joke",
#                     },
#                 },
#                 "required": ["setup", "punchline"],
#             },
#         },
#     }
# ]
# chain = prompt | model.bind_tools(tool_choice={"name": "joke"}, tools = tools)
# print(chain.invoke({"foo": "bears"}, config={}))
