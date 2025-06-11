import json
import re

from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from config.settings_config import get_settings
from agent.state import AgentState


def create_react_agent_node(tools):
    """Create a React agent node"""
    react_model = ChatOllama(
        model=get_settings().react_agent_model,
        temperature=0,
    )

    agent = create_react_agent(react_model, tools)

    async def agent_node(state: AgentState):
        result = await agent.ainvoke({"messages": state["messages"]})
        return {"messages": [result["messages"][-1]]}

    return agent_node


def generate_final_response_node(state: AgentState) -> dict:
    """Generate a structured final response"""

    # Extract conversation info
    last_message = None
    tools_used = []
    conversation_text = []

    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            conversation_text.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            conversation_text.append(f"Assistant: {msg.content}")
            last_message = msg

            # Extract tool calls if present
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if tool_call["name"] not in tools_used:
                        tools_used.append(tool_call["name"])

    # Create Ollama LLM for final response generation
    llm = ChatOllama(
        model=get_settings().final_agent_model,
        temperature=0.1,
        format="json",
    )

    # Create detailed prompt for structured response
    conversation_summary = "\n".join(conversation_text[-10:])  # Last 10 exchanges

    prompt = f"""You are tasked with creating a structured final response based on a conversation between a user and an AI assistant.

Conversation:
{conversation_summary}

Tools used during conversation: {tools_used}

Please generate a JSON response with the following structure:
{{
    "answer": "A clear, concise answer to the user's main question",
    "confidence": 0.85,
    "sources_used": ["tool1", "tool2"],
    "reasoning": "Brief explanation of how you arrived at this answer",
    "follow_up_suggestions": ["suggestion1", "suggestion2"]
}}

Requirements:
- answer: Main response to the user's question, should be comprehensive but concise
- confidence: Float between 0 and 1 indicating how confident you are in the answer
- sources_used: List the actual tools that were used (from: {tools_used})
- reasoning: Explain your reasoning process in 1-2 sentences
- follow_up_suggestions: 0-3 relevant follow-up questions the user might ask

Respond with valid JSON only:"""

    try:
        # Get response from Ollama
        response = llm.invoke(prompt)

        # Parse JSON response
        if hasattr(response, "content"):
            json_text = response.content
        else:
            json_text = str(response)

        # Clean up the JSON text (remove markdown code blocks if present)
        json_text = str(json_text)  # Ensure json_text is a string
        json_text = re.sub(r"```json\s*", "", json_text)
        json_text = re.sub(r"```\s*$", "", json_text)
        json_text = json_text.strip()

        # Parse the JSON
        final_response_dict = json.loads(json_text)

        # Validate required fields and set defaults
        final_response_dict.setdefault(
            "answer", last_message.content if last_message else "No response available"
        )
        final_response_dict.setdefault("confidence", 0.5)
        final_response_dict.setdefault("sources_used", tools_used)
        final_response_dict.setdefault(
            "reasoning", "Generated from conversation analysis"
        )
        final_response_dict.setdefault("follow_up_suggestions", [])

        # Ensure confidence is within bounds
        confidence = final_response_dict.get("confidence", 0.5)
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            final_response_dict["confidence"] = 0.5

    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"Error parsing final response: {e}")
        # Fallback response
        final_response_dict = {
            "answer": (
                last_message.content if last_message else "Unable to generate response"
            ),
            "confidence": 0.5,
            "sources_used": tools_used,
            "reasoning": "Fallback response due to parsing error",
            "follow_up_suggestions": [],
        }

    return {"structured_output": final_response_dict}
