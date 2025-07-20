import logging
from typing import Dict, List

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from core.qdrant import keyword_search_from_qdrant, search_from_qdrant

logger = logging.getLogger(__name__)


@tool
def hybrid_search_uploaded_files(
    config: RunnableConfig,
    query: str,
    file_ids: List[str],
    limit: int = 5,
    alpha: float = 0.5,
) -> Dict:
    """
    Search through uploaded files using dense semantic vector search.

    This tool performs a pure semantic search across uploaded files in the current session,
    using dense vector embeddings to find content based on meaning and context rather than
    exact keyword matches. Ideal for finding conceptually related content.

    Args:
        query (str): The search query to find semantically relevant content in uploaded files.
                    Works best with natural language questions or descriptive phrases.
        file_ids (List[str]): List of file IDs to search within. Only files with these
                             IDs will be included in the search results. Must be a
                             non-empty list of valid file identifiers.
        limit (int, optional): Maximum number of search results to return. Defaults to 5.
                              Must be a positive integer.

    Returns:
        List[Dict]: List of search results, each containing:
                   - content: The matching text/content
                   - metadata: File information and relevance scores
                   - score: Semantic similarity score for the match
                   Returns empty list if no files uploaded or no matches found.
                   Returns error dict if search fails.

    Note:
        - Uses pure semantic search (alpha=1.0) for conceptual matching
        - Best for finding content with similar meaning, even with different wording
        - Requires uploaded files to be present in the session
        - Uses Qdrant vector database for dense vector search
        - Automatically filters search to only uploaded files in current session
        - Gracefully handles errors and returns empty results if search fails
    """
    logger.debug(f"File IDs: {file_ids}")
    try:
        configurable = config.get("configurable", {})
        user_id = configurable.get("user_id")

        if not user_id:
            return {}

        final_results = {}
        for file_id in file_ids:
            results = search_from_qdrant(
                query=query,
                k=limit,
                alpha=alpha,
                metadata_filter={
                    "file_id": file_id,
                    "user_id": user_id,
                },
            )

            # Extract filename from first result's metadata (handling dict format)
            filename = None
            if results and len(results) > 0:
                # Handle both dict and Document object formats
                first_result = results[0]
                if hasattr(first_result, "metadata"):
                    # Document object format
                    filename = first_result.metadata.get(
                        "filename", f"unknown_{file_id}"
                    )
                elif isinstance(first_result, dict):
                    # Dict format - check for metadata key
                    if "metadata" in first_result:
                        filename = first_result["metadata"].get(
                            "filename", f"unknown_{file_id}"
                        )
                    else:
                        filename = f"unknown_{file_id}"
                else:
                    filename = f"unknown_{file_id}"

            # Use tuple as key: (file_id, filename)
            key = (file_id, filename)
            final_results[key] = results

        return final_results
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}


@tool
def dense_search_uploaded_files(
    config: RunnableConfig, query: str, file_ids: List[str], limit: int = 5
) -> Dict:
    """
    Search through uploaded files using dense semantic vector search.

    This tool performs a pure semantic search across uploaded files in the current session,
    using dense vector embeddings to find content based on meaning and context rather than
    exact keyword matches. Ideal for finding conceptually related content.

    Args:
        query (str): The search query to find semantically relevant content in uploaded files.
                    Works best with natural language questions or descriptive phrases.
        file_ids (List[str]): List of file IDs to search within. Only files with these
                             IDs will be included in the search results. Must be a
                             non-empty list of valid file identifiers.
        limit (int, optional): Maximum number of search results to return. Defaults to 5.
                              Must be a positive integer.

    Returns:
        List[Dict]: List of search results, each containing:
                   - content: The matching text/content
                   - metadata: File information and relevance scores
                   - score: Semantic similarity score for the match
                   Returns empty list if no files uploaded or no matches found.
                   Returns error dict if search fails.

    Note:
        - Uses pure semantic search (alpha=1.0) for conceptual matching
        - Best for finding content with similar meaning, even with different wording
        - Requires uploaded files to be present in the session
        - Uses Qdrant vector database for dense vector search
        - Automatically filters search to only uploaded files in current session
        - Gracefully handles errors and returns empty results if search fails
    """
    logger.debug(f"File IDs: {file_ids}")
    try:
        configurable = config.get("configurable", {})
        user_id = configurable.get("user_id")

        if not user_id:
            return {}

        final_results = {}
        for file_id in file_ids:
            results = search_from_qdrant(
                query=query,
                k=limit,
                alpha=1.0,
                metadata_filter={
                    "file_id": file_id,
                    "user_id": user_id,
                },
            )

            # Extract filename from first result's metadata (handling dict format)
            filename = None
            if results and len(results) > 0:
                # Handle both dict and Document object formats
                first_result = results[0]
                if hasattr(first_result, "metadata"):
                    # Document object format
                    filename = first_result.metadata.get(
                        "filename", f"unknown_{file_id}"
                    )
                elif isinstance(first_result, dict):
                    # Dict format - check for metadata key
                    if "metadata" in first_result:
                        filename = first_result["metadata"].get(
                            "filename", f"unknown_{file_id}"
                        )
                    else:
                        filename = f"unknown_{file_id}"
                else:
                    filename = f"unknown_{file_id}"

            # Use tuple as key: (file_id, filename)
            key = (file_id, filename)
            final_results[key] = results

        return final_results
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}


@tool
def sparse_search_uploaded_files(
    config: RunnableConfig, query: str, file_ids: List[str], limit: int = 5
) -> Dict:
    """
    Search through uploaded files using sparse keyword search for exact term matching.

    This tool performs a pure keyword-based search across uploaded files in the current session,
    focusing on exact term matches rather than semantic similarity. Ideal for finding specific
    words, phrases, technical terms, or when precise terminology is important.

    Args:
        query (str): The search query with specific keywords or phrases to find in uploaded files.
                    Works best with exact terms, technical terminology, or specific phrases.
        file_ids (List[str]): List of file IDs to search within. Only files with these
                             IDs will be included in the search results. Must be a
                             non-empty list of valid file identifiers.
        limit (int, optional): Maximum number of search results to return. Defaults to 5.
                              Must be a positive integer.

    Returns:
        List[Dict]: List of search results, each containing:
                   - content: The matching text/content with exact keyword matches
                   - metadata: File information and relevance scores
                   - score: Keyword matching score for the match
                   Returns empty list if no files uploaded or no matches found.
                   Returns error dict if search fails.

    Note:
        - Uses pure keyword search (alpha=0.0) for exact term matching
        - Best for finding specific terminology, proper nouns, or exact phrases
        - Does not consider semantic similarity - only exact keyword matches
        - Requires uploaded files to be present in the session
        - Uses Qdrant vector database for sparse keyword search
        - Automatically filters search to only uploaded files in current session
        - Gracefully handles errors and returns empty results if search fails
    """
    logger.debug(f"File IDs: {file_ids}")
    try:
        configurable = config.get("configurable", {})
        user_id = configurable.get("user_id")

        if not user_id:
            return {}

        final_results = {}
        for file_id in file_ids:
            results = search_from_qdrant(
                query=query,
                k=limit,
                alpha=0.0,
                metadata_filter={
                    "file_id": file_id,
                    "user_id": user_id,
                },
            )

            # Extract filename from first result's metadata (handling dict format)
            filename = None
            if results and len(results) > 0:
                # Handle both dict and Document object formats
                first_result = results[0]
                if hasattr(first_result, "metadata"):
                    # Document object format
                    filename = first_result.metadata.get(
                        "filename", f"unknown_{file_id}"
                    )
                elif isinstance(first_result, dict):
                    # Dict format - check for metadata key
                    if "metadata" in first_result:
                        filename = first_result["metadata"].get(
                            "filename", f"unknown_{file_id}"
                        )
                    else:
                        filename = f"unknown_{file_id}"
                else:
                    filename = f"unknown_{file_id}"

            # Use tuple as key: (file_id, filename)
            key = (file_id, filename)
            final_results[key] = results

        return final_results
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}


@tool
def keyword_search_from_uploaded_files(
    config: RunnableConfig, query: str, file_ids: List[str], limit: int = 5
) -> Dict:
    """
    Search through uploaded files using traditional keyword search with boolean operators.

    This tool performs a traditional keyword-based search across uploaded files in the current session,
    supporting boolean operators (AND, OR, NOT) and complex query expressions. Ideal for precise
    search queries with multiple terms and logical combinations.

    Args:
        query (str): The search query with keywords and optional boolean operators.
                    Supports boolean logic: "term1 AND term2", "term1 OR term2", "term1 NOT term2"
                    Can include exact phrases in quotes: "machine learning" AND algorithms
        file_ids (List[str]): List of file IDs to search within. Only files with these
                             IDs will be included in the search results. Must be a
                             non-empty list of valid file identifiers.
        limit (int, optional): Maximum number of search results to return. Defaults to 5.
                              Must be a positive integer.

    Returns:
        List[Dict]: List of search results, each containing:
                   - content: The matching text/content based on keyword criteria
                   - metadata: File information and relevance scores
                   - score: Keyword relevance score for the match
                   Returns empty list if no files uploaded or no matches found.
                   Returns error dict if search fails.

    Note:
        - Uses traditional keyword search with boolean operator support
        - Best for complex queries requiring logical combinations of terms
        - Supports AND, OR, NOT operators for precise search control
        - Can handle exact phrase matching with quotation marks
        - Requires uploaded files to be present in the session
        - Uses Qdrant's keyword search functionality
        - Automatically filters search to only uploaded files in current session
        - Gracefully handles errors and returns empty results if search fails
    """
    logger.debug(f"File IDs: {file_ids}")
    try:
        configurable = config.get("configurable", {})
        user_id = configurable.get("user_id")

        if not user_id:
            return {}

        final_results = {}
        for file_id in file_ids:
            results = keyword_search_from_qdrant(
                query=query,
                limit=limit,
                metadata_filter={
                    "file_id": file_id,
                    "user_id": user_id,
                },
            )

            # Extract filename from first result's metadata (handling dict format)
            filename = None
            if results and len(results) > 0:
                # Handle both dict and Document object formats
                first_result = results[0]
                if hasattr(first_result, "metadata"):
                    # Document object format
                    filename = first_result.metadata.get(
                        "filename", f"unknown_{file_id}"
                    )
                elif isinstance(first_result, dict):
                    # Dict format - check for metadata key
                    if "metadata" in first_result:
                        filename = first_result["metadata"].get(
                            "filename", f"unknown_{file_id}"
                        )
                    else:
                        filename = f"unknown_{file_id}"
                else:
                    filename = f"unknown_{file_id}"

            # Use tuple as key: (file_id, filename)
            key = (file_id, filename)
            final_results[key] = results

        return final_results

    except Exception as e:
        return {"error": f"Keyword search failed: {str(e)}"}


@tool
def compare_search_methods_for_uploaded_files(
    config: RunnableConfig, query: str, file_ids: List[str], limit: int = 3
) -> Dict[str, Dict]:
    """
    Compare multiple search methods on uploaded files using different search strategies.

    This function performs comprehensive search comparison across uploaded files in the current session,
    executing four different search strategies simultaneously to provide diverse result sets.
    Each method uses different approaches to find relevant content: semantic understanding,
    keyword matching, and hybrid combinations.

    Args:
        config (RunnableConfig): Configuration object containing uploaded file information
                                in the 'configurable' key with 'uploaded_files' list.
                                Must include valid file identifiers for search filtering.
        query (str): Search query string to find relevant content across all uploaded files.
                    Can be natural language questions, keywords, or specific terms.
        file_ids (List[str]): List of file IDs to search within. Only files with these
                             IDs will be included in the search results across all methods.
                             Must be a non-empty list of valid file identifiers.
        limit (int, optional): Maximum number of search results to return per search method.
                              Defaults to 3. Must be a positive integer.

    Returns:
        Dict[str, List[Dict]]: Dictionary containing search results for each method:
                              - "hybrid_search_uploaded_files": Combined semantic and keyword results (alpha=0.5)
                              - "dense_search_uploaded_files": Pure semantic/vector search results (alpha=1.0)
                              - "sparse_search_uploaded_files": Pure keyword-based search results (alpha=0.0)
                              - "keyword_search_from_uploaded_files": Traditional keyword search with boolean support

                              Each result contains content, metadata, and relevance scores.
                              Returns empty lists for all methods if no files uploaded.
                              Returns error dict if search operation fails.

    Note:
        - Hybrid search combines semantic understanding with keyword matching for balanced results
        - Dense search uses vector embeddings for semantic similarity and contextual understanding
        - Sparse search focuses on exact keyword matches and term frequency analysis
        - Keyword search supports boolean operators and exact phrase matching
        - All searches automatically filter to only uploaded files in current session
        - Uses Qdrant vector database for efficient search operations
        - Gracefully handles missing files and search failures
        - Alpha parameter controls the balance between dense and sparse search methods
        - Results are ranked by relevance scores specific to each search method
    """
    logger.debug(f"File IDs: {file_ids}")
    try:
        configurable = config.get("configurable", {})
        user_id = configurable.get("user_id")

        if not user_id:
            return {
                "hybrid_search_uploaded_files": {},
                "dense_search_uploaded_files": {},
                "sparse_search_uploaded_files": {},
                "keyword_search_from_uploaded_files": {},
            }

        final_results: Dict[str, Dict] = {
            "hybrid_search_uploaded_files": {},
            "dense_search_uploaded_files": {},
            "sparse_search_uploaded_files": {},
            "keyword_search_from_uploaded_files": {},
        }

        for file_id in file_ids:
            # Perform all searches for this file_id
            hybrid_results = search_from_qdrant(
                query=query,
                k=limit,
                alpha=0.5,
                metadata_filter={"file_id": file_id, "user_id": user_id},
            )
            dense_results = search_from_qdrant(
                query=query,
                k=limit,
                alpha=1.0,
                metadata_filter={"file_id": file_id, "user_id": user_id},
            )
            sparse_results = search_from_qdrant(
                query=query,
                k=limit,
                alpha=0.0,
                metadata_filter={"file_id": file_id, "user_id": user_id},
            )
            keyword_results = keyword_search_from_qdrant(
                query=query,
                limit=limit,
                metadata_filter={"file_id": file_id, "user_id": user_id},
            )

            # Extract filename from any of the results (try hybrid first, then others)
            filename = None
            for results in [
                hybrid_results,
                dense_results,
                sparse_results,
                keyword_results,
            ]:
                if results and len(results) > 0:
                    # Handle both dict and object formats
                    first_result = results[0]
                    if hasattr(first_result, "metadata"):
                        # Document object format
                        filename = first_result.metadata.get(
                            "filename", f"unknown_{file_id}"
                        )
                    elif isinstance(first_result, dict):
                        # Dictionary format - check common keys
                        filename = (
                            first_result.get("metadata", {}).get("filename")
                            or first_result.get("filename")
                            or f"unknown_{file_id}"
                        )
                    break

            # If no results found, use fallback name
            if filename is None:
                filename = f"unknown_{file_id}"

            # Create key as tuple (file_id, filename)
            key = (file_id, filename)

            # Store results with the key
            final_results["hybrid_search_uploaded_files"][key] = hybrid_results
            final_results["dense_search_uploaded_files"][key] = dense_results
            final_results["sparse_search_uploaded_files"][key] = sparse_results
            final_results["keyword_search_from_uploaded_files"][key] = keyword_results

        return final_results

    except Exception as e:
        return {"error": {"error": f"Compare search methods failed: {str(e)}"}}
