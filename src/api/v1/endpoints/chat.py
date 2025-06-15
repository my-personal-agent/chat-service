import logging

from fastapi import APIRouter, Query
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.tool import ToolMessage
from sse_starlette import EventSourceResponse

from agents.supervisor_agent import get_supervisor_agent

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/stream", response_class=EventSourceResponse)
async def chat_stream(
    message: str = Query(...),
    conversation_id: str = Query(None),
):
    agent = await get_supervisor_agent()  # type: ignore

    async def event_generator():
        config = {"configurable": {"thread_id": conversation_id}}

        is_thinking = False

        async for stream_mode, chunk in agent.astream(
            {"messages": [{"role": "user", "content": message}]},
            stream_mode=["updates", "messages"],
            config=config,  # type: ignore
        ):
            if stream_mode == "messages":
                token, _ = chunk
                if isinstance(token, AIMessageChunk):
                    if len(token.tool_calls) > 0 or len(token.tool_call_chunks) > 0:
                        logger.info(token)
                        continue

                    content = token.content

                    if content == "<think>":
                        is_thinking = True
                        yield {
                            "event": "start_thinking",
                            "data": "",
                        }
                        continue

                    if content == "</think>":
                        is_thinking = False
                        yield {
                            "event": "end_thinking",
                            "data": "",
                        }
                        continue

                    yield {
                        "event": "thinking" if is_thinking else "messaging",
                        "data": content,
                    }

                elif isinstance(token, ToolMessage):
                    logger.info(token)

        yield {
            "event": "complete",
            "data": "",
        }

    return EventSourceResponse(event_generator())
