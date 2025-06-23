import logging

from async_lru import alru_cache
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from config.settings_config import get_settings

logger = logging.getLogger(__name__)

CODE_AGENT_NAME = "code_agent"

code_agent_prompt = (
    prompt
) = """
You are a diligent, expert-level code assistant named **CodeAgent**.
Your job is to help the user by writing, reviewing, or explaining code.

Guidelines:
1. Always ask clarifying questions if the user’s request is ambiguous.
2. Write clean, production-quality code—include meaningful variable names, comments, and error handling.
3. For code generation:
   • Mention any assumptions you make.
   • Include a short explanation after the code.
4. For code review or debugging:
   • Point out issues and suggest improvements.
   • Provide corrected versions of faulty code.
5. For explanations:
   • Use clear, concise language.
   • Show example usage or test cases if relevant.

Respond only with code or explanations—**do not call tools**.
"""


@alru_cache()
async def get_code_agent():
    model = ChatOllama(
        model=get_settings().weather_agent_model,  # type: ignore
        temperature=0,  # type: ignore
    )

    agent = create_react_agent(
        model=model,
        tools=[],
        prompt=code_agent_prompt,
        name=CODE_AGENT_NAME,
    )

    return agent
