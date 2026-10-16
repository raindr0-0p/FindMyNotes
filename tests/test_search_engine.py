"""Automated tests for the semantic search engine."""

import numpy as np
import pytest

from search_engine import build_index, search_chunks


class FakeModel:
    """A deterministic fake embedding model.

    Maps each distinct text to a fixed vector. Texts not in the mapping fall
    back to a hash-derived vector, keeping results deterministic without any
    network access or real model loading.
    """

    def __init__(self, vectors: dict[str, list[float]] | None = None):
        self.vectors = dict(vectors or {})
        self.calls: list[tuple[list[str], bool]] = []

    def encode(self, texts, convert_to_numpy=True):
        self.calls.append((list(texts), bool(convert_to_numpy)))
        vectors = [self._vector(text) for text in texts]
        return np.array(vectors, dtype=np.float32)

    def _vector(self, text: str) -> np.ndarray:
        if text in self.vectors:
            return np.array(self.vectors[text], dtype=np.float32)
        # Deterministic fallback derived from the text hash, matching the
        # dimension of the provided vectors so all embeddings stay homogeneous.
        dimension = 1
        for vector in self.vectors.values():
            dimension = max(dimension, len(vector))
        seed = abs(hash(text)) % (2**31)
        rng = np.random.RandomState(seed)
        return rng.normal(size=dimension).astype(np.float32)


def make_chunks():
    return [
        {"filename": "alpha.pdf", "page": 1, "text": "alpha text"},
        {"filename": "beta.pdf", "page": 2, "text": "beta text"},
        {"filename": "gamma.pdf", "page": 3, "text": "gamma text"},
    ]


def test_build_index_from_valid_chunks():
    model = FakeModel(
        {
            "alpha text": [1.0, 0.0, 0.0],
            "beta text": [0.0, 1.0, 0.0],
            "gamma text": [0.0, 0.0, 1.0],
        }
    )
    chunks = make_chunks()
    index = build_index(chunks, model)

    assert len(index["chunks"]) == 3
    assert index["embeddings"].shape == (3, 3)
    assert len(model.calls) == 1
    assert model.calls[0][0] == [chunk["text"] for chunk in chunks]
    assert model.calls[0][1] is True


def test_build_index_preserves_chunk_metadata_and_order():
    model = FakeModel()
    chunks = make_chunks()
    index = build_index(chunks, model)

    assert [chunk["filename"] for chunk in index["chunks"]] == [
        "alpha.pdf",
        "beta.pdf",
        "gamma.pdf",
    ]
    assert [chunk["page"] for chunk in index["chunks"]] == [1, 2, 3]
    assert [chunk["text"] for chunk in index["chunks"]] == [
        "alpha text",
        "beta text",
        "gamma text",
    ]


def test_build_index_preserves_embedding_order():
    model = FakeModel(
        {
            "alpha text": [1.0, 0.0, 0.0],
            "beta text": [0.0, 1.0, 0.0],
            "gamma text": [0.0, 0.0, 1.0],
        }
    )
    index = build_index(make_chunks(), model)

    expected = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    assert np.allclose(index["embeddings"], expected)


def test_build_index_with_empty_chunk_list():
    model = FakeModel()
    index = build_index([], model)

    assert index["chunks"] == []
    assert index["embeddings"].shape == (0, 0)
    assert len(model.calls) == 0


def test_build_index_rejects_none_chunks():
    model = FakeModel()
    with pytest.raises(ValueError, match="chunks"):
        build_index(None, model)


def test_build_index_rejects_malformed_chunks():
    model = FakeModel()
    invalid_chunks = [
        {"filename": "a.pdf", "page": 1},  # missing text
        {"page": 1, "text": "hi"},  # missing filename
        {"filename": "a.pdf", "text": "hi"},  # missing page
    ]
    for chunk in invalid_chunks:
        with pytest.raises(ValueError, match="missing"):
            build_index([chunk], model)


def test_build_index_rejects_invalid_chunk_fields():
    model = FakeModel()
    bad_cases = [
        {"filename": "", "page": 1, "text": "hi"},  # empty filename
        {"filename": "   ", "page": 1, "text": "hi"},  # whitespace filename
        {"filename": "a.pdf", "page": True, "text": "hi"},  # bool page
        {"filename": "a.pdf", "page": 1.5, "text": "hi"},  # float page
        {"filename": "a.pdf", "page": 1, "text": 123},  # non-string text
    ]
    for chunk in bad_cases:
        with pytest.raises(ValueError):
            build_index([chunk], model)


def test_search_returns_empty_list_for_empty_index():
    model = FakeModel()
    index = build_index([], model)
    results = search_chunks("anything", index, model)
    assert results == []


def test_search_ranks_by_cosine_similarity():
    # orthogonal unit vectors: similarity is 1.0 only with itself, 0.0 otherwise
    model = FakeModel(
        {
            "alpha text": [1.0, 0.0],
            "beta text": [0.0, 1.0],
            "gamma text": [0.0, 1.0],
            "query text": [1.0, 0.0],
        }
    )
    chunks = make_chunks()
    index = build_index(chunks, model)

    results = search_chunks("query text", index, model, top_k=5)

    assert len(results) == 3
    assert results[0]["filename"] == "alpha.pdf"
    assert results[0]["page"] == 1
    assert results[0]["text"] == "alpha text"
    assert results[0]["similarity"] == pytest.approx(1.0)
    assert results[1]["similarity"] == pytest.approx(0.0)
    assert results[2]["similarity"] == pytest.approx(0.0)


def test_search_returns_top_k_results():
    model = FakeModel(
        {
            "alpha text": [1.0, 0.0],
            "beta text": [0.0, 1.0],
            "gamma text": [1.0, 0.0],
            "query text": [1.0, 0.0],
        }
    )
    index = build_index(make_chunks(), model)

    results = search_chunks("query text", index, model, top_k=2)

    assert len(results) == 2
    assert all(result["similarity"] == pytest.approx(1.0) for result in results)


def test_search_returns_fewer_than_top_k_when_fewer_chunks():
    model = FakeModel(
        {"alpha text": [1.0, 0.0], "query text": [1.0, 0.0]}
    )
    chunks = [{"filename": "alpha.pdf", "page": 1, "text": "alpha text"}]
    index = build_index(chunks, model)

    results = search_chunks("query text", index, model, top_k=10)

    assert len(results) == 1


def test_search_rejects_empty_query():
    model = FakeModel()
    index = build_index(make_chunks(), model)
    for bad_query in ["", "   ", "\t\n "]:
        with pytest.raises(ValueError, match="Query"):
            search_chunks(bad_query, index, model)


def test_search_rejects_non_string_query():
    model = FakeModel()
    index = build_index(make_chunks(), model)
    with pytest.raises(ValueError, match="Query"):
        search_chunks(None, index, model)


def test_search_rejects_invalid_top_k():
    model = FakeModel()
    index = build_index(make_chunks(), model)
    for bad_top_k in [0, -1, 2.5, True, "3"]:
        with pytest.raises(ValueError, match="top_k"):
            search_chunks("query", index, model, top_k=bad_top_k)


def test_search_preserves_filename_and_page_metadata():
    model = FakeModel(
        {"alpha text": [1.0, 0.0], "query text": [1.0, 0.0]}
    )
    chunks = [
        {"filename": "report.pdf", "page": 7, "text": "alpha text"},
        {"filename": "notes.pdf", "page": 2, "text": "unrelated text"},
    ]
    index = build_index(chunks, model)

    results = search_chunks("query text", index, model, top_k=5)

    assert results[0]["filename"] == "report.pdf"
    assert results[0]["page"] == 7
    assert results[0]["text"] == "alpha text"
    assert "similarity" in results[0]


def test_search_filters_by_source_filename():
    model = FakeModel(
        {
            "alpha text": [1.0, 0.0],
            "beta text": [0.0, 1.0],
            "query text": [1.0, 0.0],
        }
    )
    chunks = [
        {"filename": "alpha.pdf", "page": 1, "text": "alpha text"},
        {"filename": "beta.pdf", "page": 2, "text": "beta text"},
    ]
    index = build_index(chunks, model)

    # Query aligns with alpha (1.0), not beta (0.0); filtering to beta still
    # returns beta but with its true, unchanged similarity.
    results = search_chunks("query text", index, model, top_k=5, source_filename="beta.pdf")

    assert len(results) == 1
    assert results[0]["filename"] == "beta.pdf"
    assert results[0]["page"] == 2
    assert results[0]["text"] == "beta text"
    assert results[0]["similarity"] == pytest.approx(0.0)


def test_search_all_pdfs_option_defaults_to_all_documents():
    model = FakeModel(
        {
            "alpha text": [1.0, 0.0],
            "beta text": [0.0, 1.0],
            "query text": [1.0, 0.0],
        }
    )
    chunks = [
        {"filename": "alpha.pdf", "page": 1, "text": "alpha text"},
        {"filename": "beta.pdf", "page": 2, "text": "beta text"},
    ]
    index = build_index(chunks, model)

    results_none = search_chunks("query text", index, model, top_k=5)
    results_all_pdfs = search_chunks("query text", index, model, top_k=5)

    assert len(results_none) == 2
    assert len(results_all_pdfs) == 2
    assert results_none == results_all_pdfs


def test_search_source_filename_with_no_match_returns_empty():
    model = FakeModel(
        {"alpha text": [1.0, 0.0], "query text": [1.0, 0.0]}
    )
    chunks = [{"filename": "alpha.pdf", "page": 1, "text": "alpha text"}]
    index = build_index(chunks, model)

    results = search_chunks("query text", index, model, top_k=5, source_filename="nonexistent.pdf")

    assert results == []


def test_search_source_filename_is_case_sensitive():
    model = FakeModel(
        {"alpha text": [1.0, 0.0], "query text": [1.0, 0.0]}
    )
    chunks = [{"filename": "Alpha.pdf", "page": 1, "text": "alpha text"}]
    index = build_index(chunks, model)

    results = search_chunks("query text", index, model, top_k=5, source_filename="alpha.pdf")

    assert results == []


def test_search_source_filename_uses_stored_filename_not_reranking():
    model = FakeModel(
        {
            "alpha text": [1.0, 0.0],
            "beta text": [0.0, 1.0],
            "query text": [1.0, 0.0],
        }
    )
    chunks = [
        {"filename": "alpha.pdf", "page": 1, "text": "alpha text"},
        {"filename": "beta.pdf", "page": 2, "text": "beta text"},
    ]
    index = build_index(chunks, model)

    results = search_chunks("query text", index, model, top_k=5, source_filename="alpha.pdf")

    # Only the alpha chunk is considered, and its similarity is unchanged.
    assert len(results) == 1
    assert results[0]["filename"] == "alpha.pdf"
    assert results[0]["page"] == 1
    assert results[0]["text"] == "alpha text"
    assert results[0]["similarity"] == pytest.approx(1.0)


def test_search_rejects_invalid_source_filename():
    model = FakeModel()
    index = build_index(make_chunks(), model)
    for bad_source in ["", "   ", 123, True]:
        with pytest.raises(ValueError, match="source_filename"):
            search_chunks("query", index, model, top_k=5, source_filename=bad_source)

    # None means "all PDFs" and remains valid (default behavior).
    results = search_chunks("query", index, model, top_k=5, source_filename=None)
    assert len(results) == 3


def test_search_handles_zero_norm_query_vector():
    # zero-norm query vector must not raise; similarity treated as 0.0
    model = FakeModel(
        {"alpha text": [1.0, 0.0], "query text": [0.0, 0.0]}
    )
    index = build_index(make_chunks(), model)

    results = search_chunks("query text", index, model, top_k=5)

    assert len(results) == 3
    assert all(result["similarity"] == pytest.approx(0.0) for result in results)


def test_search_handles_zero_norm_index_vector():
    model = FakeModel(
        {
            "alpha text": [0.0, 0.0],
            "beta text": [1.0, 0.0],
            "query text": [1.0, 0.0],
        }
    )
    chunks = [
        {"filename": "alpha.pdf", "page": 1, "text": "alpha text"},
        {"filename": "beta.pdf", "page": 2, "text": "beta text"},
    ]
    index = build_index(chunks, model)

    results = search_chunks("query text", index, model, top_k=5)

    assert len(results) == 2
    # The zero-norm alpha vector yields similarity 0.0; beta yields 1.0.
    similarities = [result["similarity"] for result in results]
    assert 0.0 in similarities
    assert 1.0 in similarities


def test_search_does_not_treat_similarity_as_probability():
    # identical vectors give similarity 1.0, not a percentage
    model = FakeModel({"alpha text": [1.0, 0.0], "query text": [1.0, 0.0]})
    chunks = [{"filename": "a.pdf", "page": 1, "text": "alpha text"}]
    index = build_index(chunks, model)

    results = search_chunks("query text", index, model, top_k=5)

    assert results[0]["similarity"] == pytest.approx(1.0)


def test_model_encode_receives_convert_to_numpy_true():
    model = FakeModel()
    index = build_index(make_chunks(), model)
    search_chunks("query", index, model, top_k=5)

    assert all(convert_to_numpy for _, convert_to_numpy in model.calls)
