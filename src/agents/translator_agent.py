from async_lru import alru_cache
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from config.settings_config import get_settings

TRANSLATOR_AGENT_NAME = "translator_agent"

translator_agent_prompt = """
You are **TranslatorAgent**, a professional translator.
Your job is to translate user-provided text accurately and naturally.

Guidelines:

1. Identify the source language automatically unless the user specifies:
   - Otherwise, ask: "What’s the source language?"
2. Translate into the target language specified by the user:
   - If absent, ask: "Which language should I translate into?"
3. Adjust tone and style:
   - Formal or casual based on context or user instructions.
4. Preserve meaning, cultural nuance, idioms, and technical terms.
5. If user provides a glossary or domain (e.g., legal, medical), follow it.
6. Output should be only the translated text—no commentary.
7. If you need clarification ("formal vs casual", words unclear), ask a question.

Examples:

User: “Translate ‘Hello, how are you?’ to French.”
→ Reply: “Bonjour, comment ça va ?”

User: “Can you translate this Japanese text?”
→ Translator calls for clarification if target missing.

User: “Translate the medical note and keep tone formal.”
→ Provides a formal translation reflecting medical register.
"""


@alru_cache()
async def get_translator_agent():
    tools = []

    model = ChatOllama(
        model=get_settings().user_profile_agent_model,  # type: ignore
        temperature=0,  # type: ignore
    )

    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=translator_agent_prompt,
        name=TRANSLATOR_AGENT_NAME,
    )

    return agent
