# Requires:
# pip install langchain docarray tiktoken

# Load the API key from the .env file   从.env文件中加载API密钥
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

# define a async main function
async def main():

    vectorstore = DocArrayInMemorySearch.from_texts(
        ["harrison worked at kensho", "bears like to eat honey"],
        embedding=OpenAIEmbeddings(),
    )
    retriever = vectorstore.as_retriever()

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI()
    output_parser = StrOutputParser()

    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )
    chain = setup_and_retrieval | prompt | model | output_parser

    async for chunk in chain.astream_log(
        "where did harrison work?", include_names=["Docs"]
    ):
        print("=" * 40)
        print(chunk)

    print('-' * 40)
    async for chunk in chain.astream_log(
        "where did harrison work?", include_names=["Docs"]
    ):
        if (chunk.ops[0]["op"] == "add"):
            print(chunk.ops[0]["value"], end='', flush=True)
            await asyncio.sleep(1)

# run the async main function
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())