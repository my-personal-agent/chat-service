from async_lru import alru_cache
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_supervisor import create_supervisor

from agents.weather_agent import get_weather_agent
from config.settings_config import get_settings

SUPERVISOE_NAME = "supervisor"

checkpointer = InMemorySaver()

supervisor_prompt = """Your name is My Personal AI. Your job is to decide which agent to delegate the task to.

Use:
- `transfer_to_weather_agent` when the user asks for weather, weather forecast, air pollution.
"""


@alru_cache()
async def get_supervisor_agent():
    weather_agent = await get_weather_agent()  # type: ignore

    model = ChatOllama(
        model=get_settings().supervisor_agent_model,  # type: ignore
        temperature=0,  # type: ignore
    )

    supervisor = create_supervisor(
        agents=[weather_agent],
        model=model,
        supervisor_name=SUPERVISOE_NAME,
        prompt=supervisor_prompt,
        add_handoff_messages=False,
    ).compile(checkpointer=checkpointer)

    return supervisor
