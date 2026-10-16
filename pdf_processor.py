"""PDF text extraction and page-aware text chunking for FindMyNotes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pymupdf


def extract_pages(pdf_bytes: bytes) -> list[dict[str, Any]]:
    """Extract text from a PDF, retaining each page's 1-based page number.

    Pages with no extractable text (for example, scanned image-only pages) are
    skipped. Invalid, unreadable, or password-protected PDFs raise ``ValueError``.

    Args:
        pdf_bytes: The PDF file content as bytes.

    Returns:
        A list of dictionaries with ``page`` and ``text`` keys.
    """
    if not isinstance(pdf_bytes, (bytes, bytearray, memoryview)):
        raise ValueError("PDF content must be provided as bytes.")
    if not pdf_bytes:
        raise ValueError("PDF content is empty.")

    try:
        with pymupdf.open(stream=bytes(pdf_bytes), filetype="pdf") as document:
            if document.needs_pass:
                raise ValueError(
                    "This PDF is password-protected and cannot be read without a password."
                )

            extracted_pages: list[dict[str, Any]] = []
            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text") or ""
                text = text.strip()
                if text:
                    extracted_pages.append({"page": page_number, "text": text})

            return extracted_pages
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read PDF: {exc}") from exc


def chunk_pages(
    pages: Sequence[Mapping[str, Any]],
    filename: str,
    chunk_size: int = 200,
    overlap: int = 40,
) -> list[dict[str, Any]]:
    """Split each page into overlapping word chunks without crossing pages.

    Each returned chunk contains ``filename``, ``page``, and ``text``. A final
    chunk may contain fewer than ``chunk_size`` words. No redundant trailing
    chunk is created when the previous chunk already covers the last word.

    Args:
        pages: Page dictionaries containing a 1-based ``page`` and ``text``.
        filename: Name of the source document.
        chunk_size: Maximum number of whitespace-separated words per chunk.
        overlap: Number of words shared by consecutive chunks on the same page.

    Returns:
        A list of chunk dictionaries.

    Raises:
        ValueError: If chunking parameters or page records are invalid.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if isinstance(overlap, bool) or not isinstance(overlap, int):
        raise ValueError("overlap must be an integer.")
    if overlap < 0:
        raise ValueError("overlap cannot be negative.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("filename must be a non-empty string.")

    step = chunk_size - overlap
    chunks: list[dict[str, Any]] = []

    for page_index, page_data in enumerate(pages, start=1):
        if not isinstance(page_data, Mapping):
            raise ValueError(f"Page entry {page_index} must be a mapping.")
        if "page" not in page_data or "text" not in page_data:
            raise ValueError(
                f"Page entry {page_index} must contain 'page' and 'text' fields."
            )

        page_number = page_data["page"]
        text = page_data["text"]
        if isinstance(page_number, bool) or not isinstance(page_number, int) or page_number < 1:
            raise ValueError(f"Page entry {page_index} must have a positive integer page number.")
        if not isinstance(text, str):
            raise ValueError(f"Text for page {page_number} must be a string.")

        words = text.split()
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            if chunk_words:
                chunks.append(
                    {
                        "filename": filename,
                        "page": page_number,
                        "text": " ".join(chunk_words),
                    }
                )

            # Stop once the current chunk includes the final word. Without this
            # check, a new chunk can be created containing only already-covered
            # trailing words (for example, word 9 of a 10-word input).
            if end == len(words):
                break
            start += step

    return chunks
