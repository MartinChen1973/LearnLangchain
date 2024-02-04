# Load the API key from the .env file   从.env文件中加载API密钥
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

function = {
    "name": "solver",
    "description": "Formulates and solves an equation",
    "parameters": {
        "type": "object",
        "properties": {
            "equation": {
                "type": "string",
                "description": "The algebraic expression of the equation",
            },
            "solution": {
                "type": "string",
                "description": "The solution to the equation",
            },
        },
        "required": ["equation", "solution"],
    },
}

# Need gpt-4 to solve this one correctly
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Write out the following equation using algebraic symbols then solve it.",
        ),
        ("human", "{equation_statement}"),
    ]
)
# model = ChatOpenAI(temperature=0).bind( # gpt-3.5 cannot do it right.
model = ChatOpenAI(model="gpt-4", temperature=0).bind(
    function_call={"name": "solver"}, functions=[function]
    )
runnable = {"equation_statement": RunnablePassthrough()} | prompt | model
aiMessage = runnable.invoke("x raised to the third plus seven equals 12")
print(aiMessage)

# Get the equation and solution from the message
import json
arguments_str = aiMessage.additional_kwargs['function_call']['arguments']
arguments_dict = json.loads(arguments_str)
print(f"Equation: {arguments_dict['equation']}")
print(f"Solution: {arguments_dict['solution']}")