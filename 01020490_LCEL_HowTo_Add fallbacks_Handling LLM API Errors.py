# Load the API key from the .env file   从.env文件中加载API密钥
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain_openai import OpenAI, ChatOpenAI
from unittest.mock import patch
import httpx
from openai import RateLimitError

# Create a RateLimitError to simulate a rate limit error
request = httpx.Request("GET", "/")
response = httpx.Response(200, request=request)
error = RateLimitError("rate limit", response=response, body="")

# Note that we set max_retries = 0 to avoid retrying on RateLimits, etc
openai_chat_model = ChatOpenAI(max_retries=0, model_name="gpt-4", max_tokens=5)
openai_llm = OpenAI()
llm = openai_chat_model.with_fallbacks([openai_llm])

# Let's use just the chat model first, to show that we run into an error
with patch("openai.resources.chat.completions.Completions.create", side_effect=error):
    try:
        print(openai_chat_model.invoke("Why did the chicken cross the road?"))
    except RateLimitError:
        print("Hit error")

# Now let's try with fallbacks to llm (completion)
with patch("openai.resources.chat.completions.Completions.create", side_effect=error):
    try:
        print(llm.invoke("Why did the chicken cross the road?"))
    except RateLimitError:
        print("Hit error")