# Load the API key from the .env file   从.env文件中加载API密钥
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# define a async main function
async def main():

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai.chat_models import ChatOpenAI

    prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
    model = ChatOpenAI()
    parser = StrOutputParser()
    chain = prompt | model | parser

    async for chunk in chain.astream({"topic": "parrot"}):
        print(chunk, end="|", flush=True)     

# run the async main function
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())