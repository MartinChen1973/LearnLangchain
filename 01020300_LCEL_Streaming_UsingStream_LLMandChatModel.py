# Load the API key from the .env file   从.env文件中加载API密钥
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# define a async main function
async def main():

    from langchain_openai.chat_models import ChatOpenAI
    # from langchain.chat_models import ChatAnthropic # To use the ChatAnthropic model, you must set ANTHROPIC_API_KEY in .env. 要使用ChatAnthropic模型，您必须在.env中设置ANTHROPIC_API_KEY。

    model = ChatOpenAI()
    # model = ChatAnthropic()

    # Output with stream.
    chunks = []
    async for chunk in model.astream("hello. tell me something about yourself"):
        chunks.append(chunk)
        print(chunk.content, end="|", flush=True)

    # Chunks are additive 
    from langchain_core.messages.ai import AIMessageChunk

    full_chunk = AIMessageChunk(content="")
    for chunk in chunks:
        full_chunk = full_chunk + chunk
    print('-' * 40)
    print(full_chunk.content)

# run the async main function
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())