from langchain_community.llms import Ollama
from gen_ai_hub.proxy.langchain.init_models import init_llm


def ollama(model="gemma"):
    return Ollama(model=model)


def sapAI(model="gpt-4-32k", temperature=0):
    return init_llm(model, temperature=temperature, max_tokens=1000)
