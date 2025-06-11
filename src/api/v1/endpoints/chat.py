from fastapi import APIRouter
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage

from api.v1.models.message_request import MessageRequest
from agent.agent import get_agent

router = APIRouter()


@router.post("/{conservation_id}", response_class=JSONResponse)
async def ask(conservation_id: str, request: MessageRequest):
    agent = await get_agent()

    # Initial state
    initial_state = {
        "messages": [HumanMessage(content=request.message)],
        "structured_output": None,
    }

    # Run the graph
    result = await agent.ainvoke(initial_state)

    return result["structured_output"]
