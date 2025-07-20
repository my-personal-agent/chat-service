import logging
import re
import uuid
from typing import Any, Counter, Dict, List, Optional

from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import UpdateResult
from qdrant_client.models import (
    Condition,
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    NamedSparseVector,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
    VectorParams,
)

from config.settings_config import get_settings

logger = logging.getLogger(__name__)

client = QdrantClient(url=str(get_settings().qdrant_url))

embeddings = OllamaEmbeddings(
    model=get_settings().qdrant_embeddings_model,
    base_url=str(get_settings().ollama_base_url),
)


def _create_qdrant_collection():
    try:
        # Create collection with hybrid search support
        client.create_collection(
            collection_name=get_settings().qdrant_upload_collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=get_settings().qdrant_embeddings_model_dims,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False,
                    )
                )
            },
        )

        # Create text index for keyword search
        client.create_payload_index(
            collection_name=get_settings().qdrant_upload_collection_name,
            field_name="text",
            field_schema=TextIndexParams(
                type=TextIndexType.TEXT,
                tokenizer=TokenizerType.WORD,
                min_token_len=2,
                max_token_len=20,
                lowercase=True,
            ),
        )

        logger.info(
            f"Qdrant collection '{get_settings().qdrant_upload_collection_name}' created successfully."
        )
    except Exception as e:
        logger.error(f"Failed to create Qdrant collection: {e}")
        raise RuntimeError(f"Failed to create Qdrant collection: {e}")


def _generate_sparse_vector(
    text: str, vocabulary_size: int = 10000
) -> Dict[int, float]:
    """Generate sparse vector from text using simple term frequency approach."""
    # Simple tokenization and term frequency
    words = re.findall(r"\b\w+\b", text.lower())
    word_counts = Counter(words)

    # Convert to sparse vector (simplified approach)
    sparse_vector = {}
    for i, (word, count) in enumerate(
        word_counts.most_common(min(50, len(word_counts)))
    ):
        # Use hash of word as index (simplified)
        index = hash(word) % vocabulary_size
        sparse_vector[index] = float(count)

    return sparse_vector


def setup_qdrant():
    try:
        client.get_collection(get_settings().qdrant_upload_collection_name)
    except Exception:
        _create_qdrant_collection()


def add_documents_to_qdrant(
    documents: List[Document], ids: Optional[List[str]] = None
) -> UpdateResult:
    """
    Add documents to Qdrant collection without updating existing ones.
    Uses unique UUIDs to prevent ID conflicts.
    """
    points = []
    for i, doc in enumerate(documents):
        # Extract text and metadata from Document
        text = doc.page_content
        metadata = doc.metadata

        # Generate dense vector
        dense_vector = embeddings.embed_query(text)

        # Generate sparse vector
        sparse_vector = _generate_sparse_vector(text)

        # Create point with unique ID (UUIDs prevent conflicts)
        point_id = ids[i] if ids else str(uuid.uuid4())

        point = PointStruct(
            id=point_id,
            vector={
                "dense": dense_vector,
                "sparse": SparseVector(
                    indices=list(sparse_vector.keys()),
                    values=list(sparse_vector.values()),
                ),
            },
            payload={
                "text": text,
                "metadata": metadata,
                **metadata,  # Flatten metadata for easier filtering
            },
        )
        points.append(point)

    # Upload points - upsert with unique IDs effectively adds without updating
    result = client.upsert(
        collection_name=get_settings().qdrant_upload_collection_name, points=points
    )
    return result


def delete_documents(metadata: Dict) -> None:
    """Delete documents from Qdrant by file IDs."""

    conditions: List[Condition] = []
    for key, value in metadata.items():
        if isinstance(value, list):
            conditions.append(FieldCondition(key=key, match=MatchAny(any=value)))
        else:
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
    qdrant_filter = Filter(must=conditions)

    # Perform deletion
    client.delete(
        collection_name=get_settings().qdrant_upload_collection_name,
        points_selector=qdrant_filter,
    )


def _reciprocal_rank_fusion(dense_results, sparse_results, alpha=0.5, k=60):
    # Create score dictionaries
    dense_scores = {
        str(result.id): 1.0 / (k + i + 1) for i, result in enumerate(dense_results)
    }
    sparse_scores = {
        str(result.id): 1.0 / (k + i + 1) for i, result in enumerate(sparse_results)
    }

    # Get all unique IDs
    all_ids = set(dense_scores.keys()) | set(sparse_scores.keys())

    # Calculate combined scores
    combined_scores = {}
    for doc_id in all_ids:
        dense_score = dense_scores.get(doc_id, 0)
        sparse_score = sparse_scores.get(doc_id, 0)
        combined_scores[doc_id] = alpha * dense_score + (1 - alpha) * sparse_score

    # Sort by combined score
    sorted_ids = sorted(
        combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True
    )

    # Create result objects with combined scores
    results = []
    id_to_result = {}

    # Map results by ID
    for result in dense_results + sparse_results:
        if str(result.id) not in id_to_result:
            id_to_result[str(result.id)] = result

    # Build final results
    for doc_id in sorted_ids:
        if doc_id in id_to_result:
            result = id_to_result[doc_id]
            result.score = combined_scores[doc_id]
            results.append(result)

    return results


class QdrantResult(BaseModel):
    content: str
    metadata: Dict
    score: float
    id: str


def search_from_qdrant(
    query: str,
    k: int = 5,
    alpha: float = 0.5,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> List[QdrantResult]:
    # Generate query vectors
    dense_query = embeddings.embed_query(query)
    sparse_query = _generate_sparse_vector(query)

    # Build filter
    qdrant_filter = None
    if metadata_filter:
        conditions = []
        for key, value in metadata_filter.items():
            if isinstance(value, list):
                conditions.append(FieldCondition(key=key, match=MatchAny(any=value)))
            else:
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
        qdrant_filter = Filter(must=conditions)

    # Perform dense search
    dense_results = client.search(
        collection_name=get_settings().qdrant_upload_collection_name,
        query_vector=("dense", dense_query),
        query_filter=qdrant_filter,
        limit=k * 2,  # Get more results for fusion
        with_payload=True,
        with_vectors=False,
    )

    # Perform sparse search
    query_vector = NamedSparseVector(
        name="sparse",
        vector=SparseVector(
            indices=list(sparse_query.keys()), values=list(sparse_query.values())
        ),
    )

    sparse_results = client.search(
        collection_name=get_settings().qdrant_upload_collection_name,
        query_vector=query_vector,
        query_filter=qdrant_filter,
        limit=k * 2,
        with_payload=True,
        with_vectors=False,
    )

    # Combine results using reciprocal rank fusion
    combined_results = _reciprocal_rank_fusion(
        dense_results, sparse_results, alpha=alpha
    )

    # Format results
    formatted_results: List[QdrantResult] = []
    for result in combined_results[:k]:
        logger.info(f"Result: {result}")
        formatted_results.append(
            QdrantResult(
                content=result.payload.get("text", ""),
                metadata=result.payload.get("metadata", {}),
                score=float(result.score),
                id=result.id,
            )
        )

    return formatted_results


def keyword_search_from_qdrant(
    query: str,
    limit: int = 5,
    metadata_filter: Optional[Dict] = None,
) -> List[QdrantResult]:
    # Build filter
    qdrant_filter = None
    if metadata_filter:
        conditions = []
        for key, value in metadata_filter.items():
            if isinstance(value, list):
                conditions.append(FieldCondition(key=key, match=MatchAny(any=value)))
            else:
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
        qdrant_filter = Filter(must=conditions)

    # Perform keyword search
    results, _ = client.scroll(
        collection_name=get_settings().qdrant_upload_collection_name,
        scroll_filter=qdrant_filter,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    # Filter results by keyword presence (simple approach)
    query_words = set(query.lower().split())
    filtered_results: List[QdrantResult] = []

    for point in results:  # results is (points, next_page_offset)
        payload = point.payload if point.payload is not None else {}
        text = payload.get("text", "").lower()
        if any(word in text for word in query_words):
            filtered_results.append(
                QdrantResult(
                    content=payload.get("text", ""),
                    metadata=payload.get("metadata", {}),
                    score=1.0,
                    id=str(point.id),
                )
            )

    return filtered_results[:limit]
