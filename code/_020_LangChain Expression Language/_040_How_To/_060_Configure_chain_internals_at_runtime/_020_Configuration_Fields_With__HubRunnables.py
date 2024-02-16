# Load the API key from the .env file   从.env文件中加载API密钥
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain_core.runnables import ConfigurableField
from langchain.runnables.hub import HubRunnable

# Define a model with a configurable field
# Use a prompt defined in Hub of LangSmith rather a local one (And we get a RunnableSerializable rather than a PromptTemplate)
prompt = HubRunnable("rlm/rag-prompt").configurable_fields(
    owner_repo_commit=ConfigurableField(
        id="hub_commit",
        name="Hub Commit",
        description="The Hub commit to pull from",
    )
)

# Now using prompt from "rlm/rag-prompt"
print(prompt.invoke({"question": "foo", "context": "bar"})) 
# Swith to a different prompt from "rlm/rag-prompt-llama"
print(prompt.with_config(configurable={"hub_commit": "rlm/rag-prompt-llama"}).invoke(
    {"question": "foo", "context": "bar"}
))