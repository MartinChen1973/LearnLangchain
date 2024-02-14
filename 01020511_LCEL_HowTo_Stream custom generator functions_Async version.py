async def main():

    # Load the API key from the .env file   从.env文件中加载API密钥
    from dotenv import load_dotenv, find_dotenv
    _ = load_dotenv(find_dotenv())

    from typing import Iterator, List

    from langchain.prompts.chat import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_openai import ChatOpenAI

    prompt = ChatPromptTemplate.from_template(
        "Write a comma-separated list of 5 animals similar to: {animal}"
    )
    model = ChatOpenAI(temperature=0.0)

    str_chain = prompt | model | StrOutputParser()

    for chunk in str_chain.stream({"animal": "bear"}):
        print(chunk, end="", flush=True)
    print('\n')
    for chunk in str_chain.stream({"animal": "bear"}):
        print(chunk, end="|", flush=True)
    print('\n') 

    # This is a custom parser that splits an iterator of llm tokens
    # into a list of strings separated by commas
    from typing import AsyncIterator

    async def asplit_into_list(
        input: AsyncIterator[str],
    ) -> AsyncIterator[List[str]]:  # async def
        buffer = ""
        async for (
            chunk
        ) in input:  # `input` is a `async_generator` object, so use `async for`
            buffer += chunk
            while "," in buffer:
                comma_index = buffer.index(",")
                yield [buffer[:comma_index].strip()]
                buffer = buffer[comma_index + 1 :]
        yield [buffer.strip()]

    list_chain = str_chain | asplit_into_list

    async for chunk in list_chain.astream({"animal": "bear"}):
        print(chunk, flush=True)

# Run the main function with asyncio
import asyncio
if __name__ == "__main__":
    asyncio.run(main())