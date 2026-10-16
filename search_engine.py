"""Semantic search over indexed PDF chunks using cosine similarity."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def build_index(chunks: Sequence[Mapping[str, Any]], model: Any) -> dict[str, Any]:
    """Build an in-memory search index from chunk dictionaries.

    Each chunk must contain ``filename``, ``page``, and ``text`` keys. The
    embedding model is accepted already loaded and is never instantiated here.

    Args:
        chunks: Chunk dictionaries produced by ``pdf_processor.chunk_pages()``.
        model: An already-loaded embedding model with an ``encode`` method.

    Returns:
        A dictionary containing the chunk metadata and a NumPy matrix whose
        rows align with the input chunks in order.

    Raises:
        ValueError: If any chunk is malformed or missing required fields.
    """
    if chunks is None:
        raise ValueError("chunks must be a sequence of chunk dictionaries.")

    chunk_list = list(chunks)

    if not chunk_list:
        return {"chunks": [], "embeddings": np.empty((0, 0), dtype=np.float32)}

    texts: list[str] = []
    for chunk_index, chunk in enumerate(chunk_list, start=1):
        if not isinstance(chunk, Mapping):
            raise ValueError(f"Chunk {chunk_index} must be a mapping.")
        if "filename" not in chunk:
            raise ValueError(f"Chunk {chunk_index} is missing the 'filename' field.")
        if "page" not in chunk:
            raise ValueError(f"Chunk {chunk_index} is missing the 'page' field.")
        if "text" not in chunk:
            raise ValueError(f"Chunk {chunk_index} is missing the 'text' field.")
        if not isinstance(chunk["filename"], str) or not chunk["filename"].strip():
            raise ValueError(f"Chunk {chunk_index} has an invalid 'filename' field.")
        if isinstance(chunk["page"], bool) or not isinstance(chunk["page"], int):
            raise ValueError(f"Chunk {chunk_index} has an invalid 'page' field.")
        if not isinstance(chunk["text"], str):
            raise ValueError(f"Chunk {chunk_index} has an invalid 'text' field.")
        texts.append(chunk["text"])

    embeddings = model.encode(texts, convert_to_numpy=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    return {
        "chunks": [dict(chunk) for chunk in chunk_list],
        "embeddings": embeddings,
    }


def search_chunks(
    query: str,
    index: dict[str, Any],
    model: Any,
    top_k: int = 5,
    source_filename: str | None = None,
) -> list[dict[str, Any]]:
    """Search an index for chunks most similar to the query.

    Args:
        query: The text query to search for.
        index: An index produced by ``build_index()``.
        model: An already-loaded embedding model with an ``encode`` method.
        top_k: Maximum number of results to return (default 5).
        source_filename: If provided, restrict results to chunks whose stored
            ``filename`` matches this value exactly. The filter uses the actual
            source filename associated with each chunk and is applied before
            ranking, without re-running the embedding model or altering any
            similarity score. Use ``None`` (the default) to search all PDFs.

    Returns:
        A list of result dictionaries, each preserving ``filename``, ``page``,
        and ``text`` from the indexed chunk, plus a numerical ``similarity``
        score. Results are ranked by descending similarity.

    Raises:
        ValueError: If the query is empty/whitespace-only, if ``top_k`` is not
        a positive integer, or if ``source_filename`` is not a non-empty string.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    if source_filename is not None and (
        not isinstance(source_filename, str) or not source_filename.strip()
    ):
        raise ValueError("source_filename must be a non-empty string.")

    chunks = index.get("chunks", [])
    embeddings = index.get("embeddings", np.empty((0, 0), dtype=np.float32))

    if not chunks or embeddings.shape[0] == 0:
        return []

    if source_filename is not None:
        # Restrict candidates to the selected source PDF using the filename
        # stored with each indexed chunk. Embeddings already exist, so this
        # neither re-runs the model nor changes any similarity score.
        matching_indices = [
            i for i, chunk in enumerate(chunks) if chunk.get("filename") == source_filename
        ]
        if not matching_indices:
            return []
        chunks = [chunks[i] for i in matching_indices]
        embeddings = embeddings[np.array(matching_indices, dtype=int)]

    query_embedding = model.encode([query], convert_to_numpy=True)
    query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)

    similarities = _cosine_similarity(query_embedding, embeddings).reshape(-1)

    # Rank by descending similarity. np.argsort is stable, so ties keep their
    # original index order.
    ranked_indices = np.argsort(-similarities, kind="stable")

    results: list[dict[str, Any]] = []
    for idx in ranked_indices[:top_k]:
        source_chunk = chunks[int(idx)]
        results.append(
            {
                "filename": source_chunk["filename"],
                "page": source_chunk["page"],
                "text": source_chunk["text"],
                "similarity": float(similarities[int(idx)]),
            }
        )

    return results


def _cosine_similarity(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity between a query vector and a set of vectors.

    Zero-norm vectors are handled gracefully: a zero-norm vector yields a
    similarity of 0.0 with any other vector (including itself), avoiding
    division-by-zero errors.
    """
    query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
    embedding_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Guard against division by zero for zero-norm vectors.
    safe_query_norm = np.where(query_norm == 0, 1.0, query_norm)
    safe_embedding_norms = np.where(embedding_norms == 0, 1.0, embedding_norms)

    normalized_query = query_embedding / safe_query_norm
    normalized_embeddings = embeddings / safe_embedding_norms

    similarities = normalized_query @ normalized_embeddings.T

    # Where either vector had zero norm, the true cosine similarity is
    # undefined; treat it as 0.0 rather than the spurious value produced by
    # normalizing with the substitute norm of 1.0.
    zero_query = query_norm.reshape(-1) == 0
    zero_embedding = embedding_norms.reshape(-1) == 0
    if np.any(zero_query) or np.any(zero_embedding):
        similarities = np.where(
            zero_query.reshape(-1, 1) | zero_embedding.reshape(1, -1),
            0.0,
            similarities,
        )

    return similarities
