# Load the API key from the .env file   从.env文件中加载API密钥
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain_openai import OpenAI, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate

# First let's create a chain with a ChatModel
# We add in a string output parser here so the outputs between the two are the same type
from langchain_core.output_parsers import StrOutputParser

chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're a nice assistant who always includes a compliment in your response",
        ),
        ("human", "Why did the {animal} cross the road"),
    ]
)
# Here we're going to use a bad model name to easily create a chain that will error
chat_model = ChatOpenAI(model_name="gpt-fake")
bad_chain_with_wrong_model_name = chat_prompt | chat_model | StrOutputParser()

# Another bad chain with an invalid prompt (there is no "somewhere" in the input dict).
prompt_template = """Instructions: You should always include a compliment in your response.
Question: Why did the {animal} cross the {somewhere}?"""
prompt = PromptTemplate.from_template(prompt_template)
llm1 = OpenAI(max_tokens=1)
bad_chain_with_wrong_parameter = prompt | llm1

# Now lets create a chain with the normal OpenAI model
prompt_template = """Instructions: You should always include a compliment in your response.
Question: Why did the {animal} cross the road?"""
prompt = PromptTemplate.from_template(prompt_template)
llm2 = OpenAI()
good_chain = prompt | llm2

# We can now create a final chain which combines the two
chain = bad_chain_with_wrong_model_name.with_fallbacks([bad_chain_with_wrong_parameter, good_chain])
print(chain.invoke({"animal": "turtle"}))
