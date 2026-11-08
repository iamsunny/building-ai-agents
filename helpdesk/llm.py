"""One place to choose the model, so every part of the series can switch provider from .env."""
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

DEFAULT_MODEL = "openai:gpt-5.4-mini"


def get_model(**kwargs):
    """Return the chat model named by MODEL in .env, for example 'openai:gpt-5.4-mini'
    or 'anthropic:<model-name>'. Any chat model that supports tool calling works."""
    return init_chat_model(os.getenv("MODEL", DEFAULT_MODEL), **kwargs)
