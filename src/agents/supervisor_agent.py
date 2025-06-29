from langchain_ollama import ChatOllama
from langgraph_supervisor import create_supervisor

from agents.code_agent import get_code_agent
from agents.google_agent import get_google_agent
from agents.tools.common import calculator, get_current_time
from agents.tools.memory import manage_supervisor_memory, search_supervisor_memory
from agents.translator_agent import get_translator_agent
from agents.user_profile_agent import get_user_profile_agent
from agents.weather_agent import get_weather_agent
from config.settings_config import get_settings

SUPERVISOE_NAME = "supervisor"

# prompt
supervisor_prompt = """You are **My Personal AI**, a supervisor agent responsible for delegating user tasks to the right expert.

Your available tools:
- 🌤️ `transfer_to_weather_agent`: Use for weather queries, forecasts, air quality, or climate info.
- 👤 `transfer_to_user_profile_agent`: Use for anything involving user profile—viewing or updating.
- 💻 `transfer_to_code_agent`: Use for writing, reviewing, or explaining code.
- 🌐 `transfer_to_translator_agent`: Use for any translation requests (e.g., “Translate this to French”).
- ✉️ `transfer_to_google_agent`: Use for Gmail tasks like composing or sending emails.

⏱️ Delegation rules:
1. Do **not** respond directly to the user.
2. Choose **exactly one** tool.
"""


async def build_supervisor_agent(store, checkpointer):
    # agents
    weather_agent = await get_weather_agent()  # type: ignore
    user_profile_agent = await get_user_profile_agent()  # type: ignore
    code_agent = await get_code_agent()  # type: ignore
    translator_agent = await get_translator_agent()  # type: ignore
    google_agent = await get_google_agent()  # type: ignore

    # model
    model = ChatOllama(
        model=get_settings().supervisor_agent_model,  # type: ignore
        temperature=0,  # type: ignore
    )

    supervisor = create_supervisor(
        agents=[
            weather_agent,
            user_profile_agent,
            code_agent,
            translator_agent,
            google_agent,
        ],
        model=model,
        tools=[
            manage_supervisor_memory,
            search_supervisor_memory,
            calculator,
            get_current_time,
        ],
        supervisor_name=SUPERVISOE_NAME,
        prompt=supervisor_prompt,
        output_mode="full_history",
        add_handoff_messages=False,
    ).compile(
        checkpointer=checkpointer,
        store=store,
    )

    return supervisor
