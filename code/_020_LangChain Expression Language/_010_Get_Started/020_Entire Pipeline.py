# 0. Load the API key from the .env file 从.env文件中加载API密钥
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# What is the input of the whole chain? 整个链的输入是什么？
input = {"topic": "ice cream"}
print(f"input: {input}")

print("================== Run with steps ======================")

# What is in the output of prompt? prompt的输出是什么？
prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
promptValue = prompt.invoke(input)
print(f"prompt : {promptValue}")

# What is in the output of model? model的输出是什么？
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
baseMessage = model.invoke(promptValue)
print(f"model: {baseMessage}")

# What is in the output of output_parser? output_parser的输出是什么？
output_parser = StrOutputParser()
str = output_parser.invoke(baseMessage)
print(f"output_parser: {str}")

print("================== Run with chain ======================")
# The chain can do all the above in one line. 链可以在一行中完成上述所有操作。
chain = runnableSerializable = prompt | model | output_parser
str = chain.invoke(input)
print(f"chain: {str}")