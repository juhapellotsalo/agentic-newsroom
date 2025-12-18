import os
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

def get_mini_model(reasoning_effort: str = "minimal") -> BaseChatModel:
    return init_chat_model(model="openai:gpt-5-mini", reasoning_effort=reasoning_effort)

def get_smart_model(reasoning_effort: str = "minimal") -> BaseChatModel:
    return init_chat_model(model="openai:gpt-5", reasoning_effort=reasoning_effort)