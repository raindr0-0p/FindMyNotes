"""Opt-in end-to-end integration test using the real sentence-transformers model.

This test is skipped by default. Set ``RUN_REAL_MODEL_INTEGRATION=1`` to run it.
It loads the real ``sentence-transformers/all-MiniLM-L6-v2`` model from the local
cache (``local_files_only=True``) and exercises the complete pipeline:

PDF bytes -> extract_pages -> chunk_pages -> build_index -> search_chunks.

It does not attempt to download the model and does not modify any production code.
"""

from __future__ import annotations

import os
import pymupdf
import pytest

from pdf_processor import chunk_pages, extract_pages
from search_engine import build_index, search_chunks

RUN_REAL_MODEL = os.environ.get("RUN_REAL_MODEL_INTEGRATION", "") == "1"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QUERY = "Why does a Transformer need positional encoding to represent the order of tokens?"
FILENAME = "findmynotes.pdf"


def make_pdf_bytes() -> bytes:
    """Create a small three-page PDF entirely in memory."""
    document = pymupdf.open()
    pages = [
        (
            "Transformers, self-attention, positional encoding, and token order. "
            "A Transformer processes all tokens in parallel, so without positional "
            "encoding it cannot distinguish the order in which tokens appear."
        ),
        (
            "Photosynthesis, sunlight, water, and plants. Plants convert light "
            "energy into chemical energy using sunlight, water, and carbon dioxide."
        ),
        (
            "CPUs, instructions, registers, and arithmetic operations. A CPU fetches "
            "instructions, decodes them, and performs arithmetic on data held in registers."
        ),
    ]
    for text in pages:
        page = document.new_page()
        page.insert_text((72, 72), text)
    result = document.tobytes()
    document.close()
    return result


@pytest.mark.skipif(not RUN_REAL_MODEL, reason="Set RUN_REAL_MODEL_INTEGRATION=1 to run the real-model integration test.")
def test_real_pipeline_ranks_transformer_page_first():
    pdf_bytes = make_pdf_bytes()

    pages = extract_pages(pdf_bytes)
    assert pages, "extract_pages should return at least one page"

    chunks = chunk_pages(pages, FILENAME)
    assert chunks, "chunk_pages should return at least one chunk"

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME, local_files_only=True)

    index = build_index(chunks, model)
    assert len(index["chunks"]) == len(chunks)

    results = search_chunks(QUERY, index, model, top_k=5)
    assert results, "search_chunks should return at least one result"

    top = results[0]

    assert top["page"] == 1, (
        f"Expected the top result to come from page 1, got page {top['page']}. "
        f"Full results: {results}"
    )
    assert top["filename"] == FILENAME, (
        f"Expected filename {FILENAME!r}, got {top['filename']!r}. "
        f"Full results: {results}"
    )
    assert "positional encoding" in top["text"].lower(), (
        f"Expected top result to mention positional encoding, got: {top['text']!r}. "
        f"Full results: {results}"
    )

    # Diagnostic output for manual inspection.
    print("\nReal-model integration results:")
    for rank, result in enumerate(results, start=1):
        print(f"  {rank}. page={result['page']} similarity={result['similarity']:.4f} text={result['text']!r}")
