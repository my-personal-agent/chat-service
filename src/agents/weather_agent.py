import logging

from async_lru import alru_cache
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from config.settings_config import get_settings

logger = logging.getLogger(__name__)

WEATHER_AGENT_NAME = "weather_agent"

mcp_config = {
    "weather": {
        "url": str(get_settings().mcp_server_weather_url),
        "transport": get_settings().mcp_server_weather_transport,
    },
}

mcp_client = MultiServerMCPClient(mcp_config)  # type: ignore


@alru_cache()
async def get_weather_agent():
    tools = await mcp_client.get_tools()

    model = ChatOllama(
        model=get_settings().react_agent_model,  # type: ignore
        temperature=0,  # type: ignore
    )

    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt="You are a weather assistant",
        name=WEATHER_AGENT_NAME,
    )

    return agent
