from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI()

# import Dict
input = {
        "question": "where did he work?",
        "chat_history": [
            HumanMessage(content="Who wrote this notebook?"),
            AIMessage(content="Harrison"),
        ],
    }

#### The full chain is composed of two parts

#### First part: a context-dependent question + chat history => the standalone question
# This part replaces "he" with "Harrison" in the question, with the help of the chat history
_condense_question_prompt_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_condense_question_prompt_template)

chain_part1_condense_question = RunnablePassthrough() | CONDENSE_QUESTION_PROMPT | model | StrOutputParser()

print("## Condense question")
standalone_question = chain_part1_condense_question.invoke(input)
print(standalone_question)

#### Second part: a standalone question + context from a retriever => the final answer
# The Retriever
vectorstore = FAISS.from_texts(
    ["harrison worked at kensho"], embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

# The Prompt
_answer_prompt_template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(_answer_prompt_template)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | ANSWER_PROMPT
    | model
    | StrOutputParser()
)

# The final answer
print("## Final answer")
print(chain.invoke(standalone_question))

