from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
from llm_commons.langchain.btp_llm.openai import ChatBTPOpenAI


def ollama(model="llama2"):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    return Ollama(model=model, callback_manager=callback_manager)


def openAI(model="gpt-4-32k", temperature=0):
    return ChatBTPOpenAI(deployment_id=model, temperature=temperature)
