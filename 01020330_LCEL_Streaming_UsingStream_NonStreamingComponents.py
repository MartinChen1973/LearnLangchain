# Load the API key from the .env file   从.env文件中加载API密钥
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# define a async main function
async def main():

    from langchain_community.vectorstores import FAISS
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import OpenAIEmbeddings

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """

    vectorstore = FAISS.from_texts(
        ["harrison worked at kensho", "harrison likes spicy food"],
        embedding=OpenAIEmbeddings(),
    )
    retriever = vectorstore.as_retriever()

    chunks = [chunk for chunk in retriever.stream("where did harrison work?")]
    for chunk in chunks: # only one chunk
        print(chunk)

# run the async main function
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())