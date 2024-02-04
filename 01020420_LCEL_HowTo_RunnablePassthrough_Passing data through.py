# Load the API key from the .env file   从.env文件中加载API密钥
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Basic usage
runnable = RunnableParallel(
    passed=RunnablePassthrough(),
    extra=RunnablePassthrough.assign(mult=lambda x: x["num"] * 3),
    modified=lambda x: x["num"] + 1,
)

response = runnable.invoke({"num": 1})
print(response)

# Using Math
import math
runnable = RunnableParallel(
    num=RunnablePassthrough(),
    sqrt=RunnablePassthrough.assign(square=lambda x: math.sqrt(x["num"])),
    pow=lambda x: math.pow(2.71828, x["num"])
)

response = runnable.invoke({"num": 4})
print(response)