import logging

from async_lru import alru_cache
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

from config.settings_config import get_settings

logger = logging.getLogger(__name__)

WEATHER_AGENT_NAME = "weather_agent"

weather_agent_prompt = """
You are a knowledgeable weather assistant that provides weather information. Your capabilities include:

## Core Functions:
1. **Current Weather Analysis**: Present temperature, humidity, wind speed, conditions, and timing
2. **Air Quality Reporting**: Analyze AQI, PM2.5, PM10, O₃, NO₂ levels and provide health recommendations
3. **Weather Forecasting**: Interpret hourly/daily forecasts including temperature trends, precipitation probability, and wind patterns
4. **Weather Education**: Explain weather phenomena and pollution impacts in simple, accessible terms
5. **Comprehensive Reporting**: Always include timestamps and proper units in your responses

## Response Guidelines:
- **Natural Communication**: Present weather information conversationally, as if you're a local meteorologist providing a direct report
- **Complete Analysis**: Provide comprehensive weather summaries that include all relevant details
- **Practical Insights**: Offer advice based on conditions (clothing suggestions, activity recommendations, health considerations)
- **Clear Presentation**: Specify measurement units and timestamps naturally within your responses
- **Focused Delivery**: Present findings directly without referencing data sources, tools, or technical processes

## Information Priority:
1. Present current conditions as the primary focus
2. Highlight notable weather patterns or changes
3. Include air quality information when available
4. Provide context for unusual readings or conditions
5. Ask follow-up questions when location or time specifics are unclear

Communicate as a weather expert providing direct, useful insights about atmospheric conditions and air quality.
"""

mcp_config = {
    "weather": {
        "url": str(get_settings().mcp_server_weather_url),
        "transport": get_settings().mcp_server_weather_transport,
    },
}

mcp_client = MultiServerMCPClient(mcp_config)  # type: ignore


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
        SystemMessage(role="system", content=weather_agent_prompt)
    ]

    if last_human_message:
        messages_to_return.append(last_human_message)

    if last_message and isinstance(last_message, ToolMessage):
        messages_to_return.append(last_message)

    return messages_to_return


@alru_cache()
async def get_weather_agent() -> tuple[str, CompiledStateGraph]:
    tools = await mcp_client.get_tools()

    model = ChatOllama(
        base_url=str(get_settings().ollama_base_url),
        model=get_settings().weather_agent_model,
        temperature=0,
    )

    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=weather_agent_prompt,  # type: ignore
        name=WEATHER_AGENT_NAME,
    )

    return "Weather Agent", agent
