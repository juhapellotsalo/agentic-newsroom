from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

def get_reasoning_model() -> BaseChatModel:
    """
    Returns the flagship reasoning model for complex tasks.
    Used for: Drafting, Editing, Decision Making, Planning.
    Current: GPT-5.1 (Reasoning Effort: Medium)
    """
    return init_chat_model(model="openai:gpt-5.1", reasoning_effort="medium")

def get_mini_model() -> BaseChatModel:
    """
    Returns a faster, smaller model for routine tasks.
    Used for: Simple formatting, basic extractions, high-volume classifications.
    Current: GPT-4o-mini
    """
    return init_chat_model(model="openai:gpt-4o-mini")
