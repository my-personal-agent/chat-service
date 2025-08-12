from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer
from langgraph_supervisor import create_supervisor

from agents.code_agent import get_code_agent
from agents.google_agent import get_google_agent
from agents.tools.common import calculator, get_current_time
from agents.translator_agent import get_translator_agent
from agents.upload_file_agent import get_upload_file_agent
from agents.user_profile_agent import get_user_profile_agent
from agents.weather_agent import get_weather_agent
from config.settings_config import get_settings

SUPERVISOR_NAME = "supervisor"

# prompt
supervisor_prompt = """
You are a routing agent that directs user requests to the appropriate specialized agents based on request type and intent.

## Available Specialized Agents

**Weather Agent**: Current conditions, forecasts, temperature, precipitation, weather alerts, climate data

## Routing Decision Process

1. **Analyze the user's request** to identify the primary intent and domain
2. **Match to available agents** based on their capabilities
3. **Route to appropriate agent** using the corresponding transfer function
4. **If no specialized agent matches**, provide a direct helpful response

## Routing Guidelines

### Weather Queries → `transfer_to_weather_agent`
- Weather conditions, forecasts, temperature queries
- Precipitation, storms, climate information
- Weather planning, alerts, location-based weather
- Keywords: weather, temperature, forecast, rain, snow, sunny, cloudy, storm, etc.

### General Queries → Direct Response
For requests outside specialized agent domains:
1. Provide helpful information directly
2. Mention relevant available services
3. Suggest which specialized agent might help if applicable

## Core Operating Rules

- **Route first, answer second**: If a request matches a specialized agent, transfer immediately
- **Single-agent routing**: Route to the most appropriate agent, don't attempt multiple transfers
- **Context preservation**: Include relevant context when transferring to agents
- **Graceful fallback**: For unmatched requests, respond helpfully and mention available services

## Response Templates

**Successful Route**: Transfer immediately with context
**No Match**: Direct response + "I have specialized agents available for [list relevant services]"
**Unclear Intent**: Ask for clarification + mention available specialized services
"""


def prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
    last_message = state["messages"][-1] if state["messages"] else None

    # Find the last HumanMessage
    last_human_message = (
        next(
            (
                msg
                for msg in reversed(state["messages"])
                if isinstance(msg, HumanMessage)
            ),
            None,
        )
        if state["messages"]
        else None
    )

    messages_to_return: list[AnyMessage] = [
        SystemMessage(role="system", content=supervisor_prompt)
    ]

    if last_human_message:
        messages_to_return.append(last_human_message)

    if last_message and isinstance(last_message, AIMessage):
        messages_to_return.append(last_message)

    return messages_to_return


async def build_supervisor_agent(
    store: BaseStore, checkpointer: Checkpointer
) -> tuple[CompiledStateGraph, dict[str, str], dict[str, list[str]]]:
    # agents
    weather_agent_name, weather_agent = await get_weather_agent()
    user_profile_agent_name, user_profile_agent = await get_user_profile_agent()
    code_agent_name, code_agent = await get_code_agent()
    translator_agent_name, translator_agent = await get_translator_agent()
    google_agent_name, google_agent, google_confirm_tools = await get_google_agent()
    upload_file_agent_name, upload_file_agent = await get_upload_file_agent()

    # model
    model = ChatOllama(
        base_url=str(get_settings().ollama_base_url),
        model=get_settings().supervisor_agent_model,
        temperature=0,
    )

    supervisor = create_supervisor(
        agents=[
            weather_agent,
            user_profile_agent,
            code_agent,
            translator_agent,
            google_agent,
            upload_file_agent,
        ],
        model=model,
        tools=[calculator, get_current_time],
        supervisor_name=SUPERVISOR_NAME,
        prompt=prompt,  # type: ignore
        output_mode="last_message",
        add_handoff_messages=False,
    ).compile(
        checkpointer=checkpointer,
        store=store,
    )

    agent_names = {
        SUPERVISOR_NAME: "Supervisor",
        weather_agent.name: weather_agent_name,
        user_profile_agent.name: user_profile_agent_name,
        code_agent.name: code_agent_name,
        translator_agent.name: translator_agent_name,
        google_agent.name: google_agent_name,
        upload_file_agent.name: upload_file_agent_name,
    }
    confirm_tools = {google_agent.name: google_confirm_tools}

    return supervisor, agent_names, confirm_tools
