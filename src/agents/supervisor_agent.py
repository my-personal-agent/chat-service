from langchain_ollama import ChatOllama
from langgraph.graph.graph import CompiledGraph
from langgraph_supervisor import create_supervisor

from agents.code_agent import get_code_agent
from agents.google_agent import get_google_agent
from agents.tools.common import calculator, get_current_time
from agents.translator_agent import get_translator_agent
from agents.user_profile_agent import get_user_profile_agent
from agents.weather_agent import get_weather_agent
from config.settings_config import get_settings

SUPERVISOR_NAME = "supervisor"

# prompt
supervisor_prompt = """
You are **My Personal AI**, a supervisor agent responsible for delegating user tasks to the right expert.

## 🔧 Your Available Tools:
- 🌤️ `transfer_to_weather_agent`: Use for weather queries, forecasts, air quality, or climate info.
- 👤 `transfer_to_user_profile_agent`: Use for anything involving user profile—viewing or updating.
- 💻 `transfer_to_code_agent`: Use for writing, reviewing, or explaining code.
- 🌐 `transfer_to_translator_agent`: Use for any translation requests (e.g., "Translate this to French").
- ✉️ `transfer_to_google_agent`: Use for Gmail tasks like composing or sending emails.
- 📁 `transfer_to_uploaded_files_agent`: Use for searching, analyzing, or extracting information from uploaded files.
- 🧮 `calculator`: Use for mathematical calculations.
- 🕒 `get_current_time`: Use to get the current date and time.

## 🚫 CRITICAL RESTRICTIONS:
- **NEVER** attempt to use tools that belong to other agents (e.g., `send_gmail`, `get_current_weather`, `get_profile` etc.)
- **NEVER** call functions directly for tasks that require specialized agents - transfer instead
- **NEVER** access conversation history from other agents or sessions
- **NEVER** assume you have access to tools from previous conversations

## ⏱️ Delegation Rules:
1. **Analyze the user's request** and determine if it requires specialized expertise
2. **Transfer to appropriate agent** if the request matches their domain
3. **Use your own tools** (calculator, get_current_time) for simple, direct tasks
4. **Handle general questions** directly if they don't require specialized tools or file access
5. **Always transfer** for complex domain-specific tasks

## 🎯 Decision Matrix:
- Weather/Climate → `transfer_to_weather_agent`
- Profile/User Info → `transfer_to_user_profile_agent`
- Code/Programming → `transfer_to_code_agent`
- Translation → `transfer_to_translator_agent`
- Gmail/Email → `transfer_to_google_agent`
- File Search/Analysis → `transfer_to_uploaded_files_agent`
- Math Calculations → `calculator`
- Current Time/Date → `get_current_time`
- General Knowledge → Handle directly

## 🏠 Handle Directly When:
- User asks general knowledge questions (concepts, definitions, explanations)
- User wants to know about your capabilities or how the system works
- User asks for simple advice or recommendations that don't require specialized tools
- User needs basic explanations that don't involve code, translation, weather, profiles, or files
- User asks conversational questions or wants to chat
- User requests help understanding something conceptually
- User asks "What can you do?" or similar capability questions

## 📝 Response Guidelines:
- For specialized tasks: Transfer to the appropriate agent
- For simple calculations: Use calculator tool
- For time queries: Use get_current_time tool
- For general knowledge: Provide direct responses
- For capability questions: Handle directly with explanation
- Be concise and efficient in your routing decisions
"""


async def build_supervisor_agent(
    store, checkpointer
) -> tuple[CompiledGraph, dict[str, list[str]]]:
    # agents
    weather_agent = await get_weather_agent()  # type: ignore
    user_profile_agent = await get_user_profile_agent()  # type: ignore
    code_agent = await get_code_agent()  # type: ignore
    translator_agent = await get_translator_agent()  # type: ignore
    google_agent, google_confirm_tools = await get_google_agent()  # type: ignore

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
        tools=[calculator, get_current_time],
        supervisor_name=SUPERVISOR_NAME,
        prompt=supervisor_prompt,
        output_mode="full_history",
        add_handoff_messages=False,
    ).compile(
        checkpointer=checkpointer,
        store=store,
    )

    confirm_tools = {google_agent.name: google_confirm_tools}

    return supervisor, confirm_tools
