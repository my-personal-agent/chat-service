import logging

from async_lru import alru_cache
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from config.settings_config import get_settings

logger = logging.getLogger(__name__)

WEATHER_AGENT_NAME = "weather_agent"

weather_agent_prompt = """
You are a knowledgeable weather assistant. You can:
1. Provide current weather (temperature, humidity, wind speed, conditions, timestamp).
2. Report air quality (AQI, PM2.5, PM10, O₃, NO₂) with health tips.
3. Give short-term forecast (hourly/daily), covering temperature, precipitation, wind, AQ trends.
4. Interpret and explain weather or pollution issues simply.
5. Always cite data timestamp and units.
6. Ask follow-up questions on ambiguity.
Respond conversationally.
"""

mcp_config = {
    "weather": {
        "url": str(get_settings().mcp_server_weather_url),
        "transport": get_settings().mcp_server_weather_transport,
    },
}

mcp_client = MultiServerMCPClient(mcp_config)  # type: ignore


@alru_cache()
async def get_weather_agent() -> CompiledStateGraph:
    tools = await mcp_client.get_tools()

    model = ChatOllama(
        model=get_settings().weather_agent_model,  # type: ignore
        temperature=0,  # type: ignore
    )

    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=weather_agent_prompt,
        name=WEATHER_AGENT_NAME,
    )

    return agent
