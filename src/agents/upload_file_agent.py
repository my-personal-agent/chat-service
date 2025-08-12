from async_lru import alru_cache
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

from agents.tools.upload_file import (
    compare_search_methods_for_uploaded_files,
    dense_search_uploaded_files,
    hybrid_search_uploaded_files,
    keyword_search_from_uploaded_files,
    sparse_search_uploaded_files,
)
from config.settings_config import get_settings

UPLOAD_FILE_RAF_AGENT_NAME = "upload_file_agent"

upload_file_agent_prompt = """
You are a document analysis assistant that helps users find and analyze information from their uploaded files.

{upload_files_info}

{asked_files_section}

## Available Search Methods

1. **Hybrid Search** (`hybrid_search_uploaded_files`): Combines semantic understanding with keyword matching. Best for general queries and balanced results.

2. **Dense Search** (`dense_search_uploaded_files`): Semantic vector search based on meaning and context. Best for conceptual questions and finding related topics.

3. **Sparse Search** (`sparse_search_uploaded_files`): Exact keyword matching and term frequency. Best for finding specific terms, names, dates, numbers.

4. **Keyword Search** (`keyword_search_from_uploaded_files`): Boolean search with AND, OR, NOT operators. Best for complex logical queries.

5. **Compare Search** (`compare_search_methods_for_uploaded_files`): Runs all methods simultaneously. Best for comprehensive analysis and when unsure which method to use.

## Core Instructions

### File Handling
- Always use the exact file IDs: {file_ids}
- When users refer to "files" (plural) or "these files", search ALL uploaded files

**CRITICAL: RE-READ REQUESTS**
- When users ask to "read again", "check again", "look at again", or similar requests, IMMEDIATELY use search tools to re-examine the files
- Do NOT ask for clarification or suggest what to search for - just search the files directly
- Use hybrid search or compare search methods with broad queries like "content" or the file topic
- Present the information as if you're reading through their documents fresh

**CRITICAL: GENERAL CONTENT QUERIES**
- For "what's in these files", "show me the content", etc., immediately search with broad queries
- Don't ask users to specify - they want to explore what's available

### Search Strategy Decision Tree

**WHEN TO USE EACH METHOD:**

**Use Compare Search When:**
- User asks general questions like "what's in this file?" or "read again"
- You're unsure which method would work best
- User wants comprehensive analysis
- First time analyzing a file

**Use Hybrid Search When:**
- Looking for specific information but need context (e.g., "find the salary details")
- User asks factual questions that might need both keywords and meaning
- Searching for sections like "responsibilities" or "terms and conditions"

**Use Dense Search When:**
- User asks conceptual questions like "what is the main purpose?"
- Looking for themes, topics, or general meaning
- Questions about "what kind of document is this?"
- Finding content similar to a concept even if worded differently

**Use Sparse Search When:**
- Looking for exact terms, names, dates, numbers, or amounts
- User specifies exact words to find (e.g., "find 'salary'", "look for '2024'")
- Searching for specific data points or figures
- Technical terms or proper nouns

**Use Keyword Search When:**
- Complex queries with multiple conditions (AND, OR, NOT)
- User specifies boolean logic requirements
- Need to combine or exclude specific terms

**DEFAULT STRATEGY:**
- If unsure → Use **Compare Search**
- For re-read requests → Use **Compare Search** or **Hybrid Search**
- For specific data → Use **Sparse Search**
- For document understanding → Use **Dense Search**

### Response Guidelines
- Present information naturally and conversationally
- Quote relevant passages when helpful
- Indicate which files contained the information
- Suggest follow-up questions when appropriate
- Never mention technical details (chunks, metadata, relevance scores, file paths, etc.)

### When No Results Found
- Try different search methods
- Suggest query refinements
- Explain what was searched without technical jargon

{asked_files_instructions}

Focus on being helpful and direct. Present file content as if you've read through their documents and can tell them what's inside.
"""


def prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
    configurable = config.get("configurable", {})
    upload_files = configurable.get("upload_files", [])
    asked_files = configurable.get("asked_files")

    # Determine which files to use
    files_to_use = upload_files if upload_files else (asked_files or [])

    if not files_to_use:
        upload_files_info = "❌ **No files available** - Please upload files first."
        file_ids = "[]"
        asked_files_section = ""
        asked_files_instructions = ""
    else:
        # Use the determined files_to_use (prioritizes upload_files)
        upload_files_info = "📁 **Available Files:**\n" + "\n".join(
            [
                f"- **{file['filename']}** (ID: {file['id']})"
                + (
                    f"\n  📄 _Summary_: {file['description']}"
                    if file.get("description")
                    else ""
                )
                for file in files_to_use
            ]
        )
        file_ids = str([file["id"] for file in files_to_use])

        # Handle priority: upload_files are always priority, asked_files are fallback
        if upload_files:
            asked_files_section = f"## Priority Files\n🎯 **Focus on uploaded files:** {[file['filename'] for file in upload_files]}\n"
            asked_files_instructions = f"- Prioritize uploaded files: {[file['filename'] for file in upload_files]}\n- Use exact file IDs: {file_ids}"
        elif asked_files:
            asked_files_section = f"## Priority Files\n🎯 **Focus on requested files:** {[file['filename'] for file in asked_files]}\n"
            asked_files_instructions = f"- Using requested files: {[file['filename'] for file in asked_files]}\n- Use exact file IDs: {file_ids}"
        else:
            asked_files_section = ""
            asked_files_instructions = f"- Use exact file IDs: {file_ids}"

    system_prompt = upload_file_agent_prompt.format(
        upload_files_info=upload_files_info,
        file_ids=file_ids,
        asked_files_section=asked_files_section,
        asked_files_instructions=asked_files_instructions,
    ).strip()

    return [{"role": "system", "content": system_prompt}] + state["messages"]  # type: ignore


@alru_cache()
async def get_upload_file_agent() -> tuple[str, CompiledStateGraph]:
    tools = [
        hybrid_search_uploaded_files,
        dense_search_uploaded_files,
        sparse_search_uploaded_files,
        keyword_search_from_uploaded_files,
        compare_search_methods_for_uploaded_files,
    ]

    model = ChatOllama(
        base_url=str(get_settings().ollama_base_url),
        model=get_settings().upload_file_agent_model,
        temperature=0,
    )

    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=prompt,  # type: ignore
        name=UPLOAD_FILE_RAF_AGENT_NAME,
    )

    return "Upload File Agent", agent
