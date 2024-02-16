# Load the API key from the .env file   从.env文件中加载API密钥
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI()
# model = ChatAnthropic()

# Define a chain to find which branch chain should be used
find_model_chain = (
    PromptTemplate.from_template(
        """Given the user question below, classify it as either being about `LangChain`, `Anthropic`, or `General`.

Do not respond with more than one word.

<question>
{question}
</question>

Classification:"""
    )
    | model
    | StrOutputParser()
)

# Define branch chains
anthropic_chain = (
    PromptTemplate.from_template(
        """You are an expert in anthropic. \
Always answer questions starting with "(Anthropic) As Dario Amodei told me". \
Respond to the following question:

Question: {question}
Answer:"""
    )
    | model
)
langchain_chain = (
    PromptTemplate.from_template(
        """You are an expert in langchain. \
Always answer questions starting with "(Langchain) As Harrison Chase told me". \
Respond to the following question:

Question: {question}
Answer:"""
    )
    | model
)
general_chain = (
    PromptTemplate.from_template(
        """Respond to the following question:

Question: {question}
Answer:"""
    )
    | model
)


# Define a branch chain
from langchain_core.runnables import RunnableBranch
branch_chain = RunnableBranch(
    (lambda x: "anthropic" in x["topic"].lower(), anthropic_chain),
    (lambda x: "langchain" in x["topic"].lower(), langchain_chain),
    general_chain,
) | StrOutputParser()

# Combine the chains into a full chain
full_chain = {"topic": find_model_chain, "question": lambda x: x["question"]} | branch_chain

# Invoke the branch chain
result = full_chain.invoke({"question": "how do I use Anthropic?"})
print(result)
print('-' * 40)

result = full_chain.invoke({"question": "how do I use LangChain?"})
print(result)
print('-' * 40)

result = full_chain.invoke({"question": "whats 2 + 2"})
print(result)