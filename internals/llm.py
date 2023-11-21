# langchain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama

# sap llm
from llm_commons.proxy.base import set_proxy_version
from llm_commons.langchain.proxy import init_llm, init_embedding_model

set_proxy_version("btp")
# set_proxy_version('aicore') # for an AI Core proxy


def ollama(model="llama2"):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    return Ollama(model=model, callback_manager=callback_manager)


# https://github.tools.sap/AI-Playground-Projects/llm-commons
def sapAI(model="gpt-4", temperature=0):
    return init_llm(model, temperature=temperature, max_tokens=8000)
