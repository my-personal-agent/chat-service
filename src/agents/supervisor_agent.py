from langchain_ollama import ChatOllama
from langgraph_supervisor import create_supervisor

from agents.tools.memory import manage_supervisor_memory, search_supervisor_memory
from agents.weather_agent import get_weather_agent
from config.settings_config import get_settings

SUPERVISOE_NAME = "supervisor"

# prompt
supervisor_prompt = """Your name is My Personal AI. Your job is to decide which agent to delegate the task to.

Use:
- `transfer_to_weather_agent` when the user asks for weather, weather forecast, air pollution.
"""


async def build_supervisor_agent(store, checkpointer):
    # agents
    weather_agent = await get_weather_agent()  # type: ignore

    # model
    model = ChatOllama(
        model=get_settings().supervisor_agent_model,  # type: ignore
        temperature=0,  # type: ignore
    )

    supervisor = create_supervisor(
        agents=[weather_agent],
        model=model,
        tools=[manage_supervisor_memory, search_supervisor_memory],
        supervisor_name=SUPERVISOE_NAME,
        prompt=supervisor_prompt,
        output_mode="full_history",
        add_handoff_messages=False,
    ).compile(
        checkpointer=checkpointer,
        store=store,
    )

    return supervisor
