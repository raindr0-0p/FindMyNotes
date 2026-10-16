import pytest
import pymupdf

from pdf_processor import chunk_pages, extract_pages


def make_pdf_bytes(page_texts):
    """Create a small PDF in memory; None or an empty string makes a blank page."""
    document = pymupdf.open()
    for text in page_texts:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
    result = document.tobytes()
    document.close()
    return result


def test_extract_pages_from_valid_pdf_and_keep_one_based_page_numbers():
    pdf_bytes = make_pdf_bytes(["First page text", "Second page text"])

    assert extract_pages(pdf_bytes) == [
        {"page": 1, "text": "First page text"},
        {"page": 2, "text": "Second page text"},
    ]


def test_extract_pages_rejects_invalid_pdf():
    with pytest.raises(ValueError, match="Could not read PDF|empty"):
        extract_pages(b"not a valid PDF")


def test_extract_pages_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        extract_pages(b"")


def test_extract_pages_skips_pages_without_extractable_text():
    pdf_bytes = make_pdf_bytes(["Text on page one", "", "Text on page three"])

    assert extract_pages(pdf_bytes) == [
        {"page": 1, "text": "Text on page one"},
        {"page": 3, "text": "Text on page three"},
    ]


def test_chunk_pages_basic_overlapping_chunks_without_redundant_tail():
    words = [f"word{i}" for i in range(10)]
    pages = [{"page": 1, "text": " ".join(words)}]

    chunks = chunk_pages(pages, "notes.pdf", chunk_size=5, overlap=2)

    assert [chunk["text"].split() for chunk in chunks] == [
        words[0:5],
        words[3:8],
        words[6:10],
    ]
    assert len(chunks) == 3


def test_chunks_preserve_filename_and_page_metadata():
    pages = [{"page": 4, "text": "one two three four five six"}]

    chunks = chunk_pages(pages, "lecture.pdf", chunk_size=4, overlap=1)

    assert all(chunk["filename"] == "lecture.pdf" for chunk in chunks)
    assert all(chunk["page"] == 4 for chunk in chunks)


def test_pages_are_chunked_independently():
    pages = [
        {"page": 1, "text": "a b c d e f"},
        {"page": 2, "text": "g h i j k l"},
    ]

    chunks = chunk_pages(pages, "notes.pdf", chunk_size=4, overlap=1)

    assert [chunk["page"] for chunk in chunks] == [1, 1, 2, 2]
    assert chunks[1]["text"] == "d e f"
    assert chunks[2]["text"] == "g h i j"
    assert all(not ({"f", "g"} <= set(chunk["text"].split())) for chunk in chunks)


def test_empty_page_text_produces_no_empty_chunks():
    pages = [{"page": 1, "text": "   \n  "}, {"page": 2, "text": ""}]

    assert chunk_pages(pages, "empty.pdf") == []


def test_invalid_chunking_parameters_are_rejected():
    pages = [{"page": 1, "text": "one two three"}]
    invalid_options = [
        {"chunk_size": 0, "overlap": 0},
        {"chunk_size": -1, "overlap": 0},
        {"chunk_size": 5, "overlap": -1},
        {"chunk_size": 5, "overlap": 5},
        {"chunk_size": 5, "overlap": 6},
    ]

    for options in invalid_options:
        with pytest.raises(ValueError):
            chunk_pages(pages, "notes.pdf", **options)


def test_chunking_rejects_non_integer_parameters():
    pages = [{"page": 1, "text": "one two three"}]

    with pytest.raises(ValueError, match="chunk_size"):
        chunk_pages(pages, "notes.pdf", chunk_size=2.5, overlap=0)
    with pytest.raises(ValueError, match="overlap"):
        chunk_pages(pages, "notes.pdf", chunk_size=3, overlap=1.5)
