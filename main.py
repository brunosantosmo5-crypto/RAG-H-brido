import os
import langchain as lc
import pandas as pd
import chainlit as cl
from dotenv import load_dotenv
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openrouter import OpenRouter

load_dotenv()

SYSTEM_PROMPT = """
Você é um assistente técnico direto, preciso e útil.
Responda sempre em português do Brasil.
Quando não souber algo, diga que não sabe.
"""

def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"A variável de ambiente {name} não foi configurada.")
    return value

def create_llm() -> OpenRouter:
    return OpenRouter(
        model=os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct"),
        api_key=get_required_env("OPENROUTER_API_KEY"),
        context_window=8192,
        max_tokens=512,
        temperature=0.1,
    )