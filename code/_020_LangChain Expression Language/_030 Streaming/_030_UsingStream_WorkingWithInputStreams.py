# Load the API key from the .env file   从.env文件中加载API密钥
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# define a async main function
async def main():

    from langchain_core.output_parsers import JsonOutputParser
    from langchain_openai.chat_models import ChatOpenAI

    model = ChatOpenAI()

    chain = model | JsonOutputParser()  # This parser only works with OpenAI right now
    async for text in chain.astream(
        'output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`'
    ):
        print(text, flush=True) 

# run the async main function
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())