# long-term memory and semantic search

from langmem import create_manage_memory_tool, create_search_memory_tool

manage_supervisor_memory = create_manage_memory_tool(
    namespace=("memories", "{user_id}", "{chat_id}"),
)
search_supervisor_memory = create_search_memory_tool(
    namespace=("memories", "{user_id}", "{chat_id}")
)
