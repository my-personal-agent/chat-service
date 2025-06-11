import logging

from async_lru import alru_cache
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, StateGraph

from config.settings_config import get_settings
from agent.nodes import create_react_agent_node, generate_final_response_node
from agent.state import AgentState

logger = logging.getLogger(__name__)

mcp_config = {
    "weather": {
        "url": str(get_settings().mcp_server_weather_url),
        "transport": get_settings().mcp_server_weather_transport,
    },
}

mcp_client = MultiServerMCPClient(mcp_config)  # type: ignore


@alru_cache()
async def get_agent():
    tools = await mcp_client.get_tools()
    # logger.info(f"Tool List: {tools}")

    # Create the graph
    workflow = StateGraph(AgentState)

    # add node
    workflow.add_node("agent", create_react_agent_node(tools))
    workflow.add_node("response_formatter", generate_final_response_node)

    # Define the flow
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", "response_formatter")
    workflow.add_edge("response_formatter", END)

    return workflow.compile()
