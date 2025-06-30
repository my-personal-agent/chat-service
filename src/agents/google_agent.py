import logging

from async_lru import alru_cache
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

from config.settings_config import get_settings

logger = logging.getLogger(__name__)

GOOGLE_AGENT_NAME = "google_agent"

google_agent_prompt = """
You are the **Google Assistant**, a versatile AI agent empowered to help users manage their Google tasks. {user_fullname}
{google_ids}
### Available tools:
- 📨 `send_gmail(gmail_user_id, to, subject, body)`
  - `gmail_user_id` (string): the user’s Gmail token ID, retrieved from the Runnable Configuration (do **not** ask the user to provide it)
  - `to` (string, valid email)
  - `subject` (string, non-empty, ≤ 998 chars)
  - `body` (string, non-empty, supports plain-text or HTML)

### Email Formatting Instructions:
{email_sign}- Maintain professional email formatting and tone
"""


def prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
    configurable = config.get("configurable", {})

    # 1. Personalized greeting
    user_fullname = configurable.get("user_fullname")
    fullname_section = (
        f"Address the user as **{user_fullname}**." if user_fullname else ""
    )
    email_sign_section = (
        f"- Always sign emails with the user's full name: **{fullname_section}**\n"
        if user_fullname
        else ""
    )

    # google
    google_ids = []

    # gmail
    gmail_user_id = configurable.get("gmail_user_id")
    if gmail_user_id:
        google_ids.append(
            f"Your `gmail_user_id` is `{gmail_user_id}` and should be used automatically when calling the send_gmail tool."
        )

    if len(google_ids) > 0:
        google_ids_section = "\n".join(google_ids)
    else:
        google_ids_section = ""

    system_prompt = google_agent_prompt.format(
        user_fullname=fullname_section,
        google_ids=google_ids_section,
        email_sign=email_sign_section,
    ).strip()

    return [{"role": "system", "content": system_prompt}] + state["messages"]  # type: ignore


mcp_config = {
    "google": {
        "url": str(get_settings().mcp_server_google_url),
        "transport": get_settings().mcp_server_google_transport,
    },
}

mcp_client = MultiServerMCPClient(mcp_config)  # type: ignore


@alru_cache()
async def get_google_agent():
    tools = await mcp_client.get_tools()

    model = ChatOllama(
        model=get_settings().google_agent_model,  # type: ignore
        temperature=0,  # type: ignore
    )

    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=prompt,  # type: ignore
        name=GOOGLE_AGENT_NAME,
        interrupt_before=["tools"],
    )

    return agent
