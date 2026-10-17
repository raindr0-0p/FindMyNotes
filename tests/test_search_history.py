"""Tests for the search history helpers."""

import pytest

from search_history import MAX_HISTORY_ENTRIES, add_history_entry, clear_history


def test_history_records_most_recent_first():
    history = []
    history = add_history_entry(history, "first query", "All PDFs", 5, 3)
    history = add_history_entry(history, "second query", "beta.pdf", 5, 1)
    history = add_history_entry(history, "third query", "All PDFs", 3, 0)

    assert [entry["query"] for entry in history] == [
        "third query",
        "second query",
        "first query",
    ]


def test_history_limits_to_ten_entries():
    history = []
    for i in range(15):
        history = add_history_entry(
            history, f"query {i}", "All PDFs", 5, i
        )

    assert len(history) == MAX_HISTORY_ENTRIES
    assert [entry["query"] for entry in history] == [
        f"query {i}" for i in range(14, 4, -1)
    ]


def test_history_stores_query_source_filter_result_count_and_num_results():
    history = add_history_entry(
        [], "query text", "alpha.pdf", 10, 2
    )

    assert history[0]["query"] == "query text"
    assert history[0]["source_filter"] == "alpha.pdf"
    assert history[0]["result_count"] == 10
    assert history[0]["num_results"] == 2
    assert "timestamp" in history[0]


def test_history_records_empty_result_searches():
    history = add_history_entry([], "no matches here", "All PDFs", 5, 0)

    assert len(history) == 1
    assert history[0]["query"] == "no matches here"
    assert history[0]["num_results"] == 0


def test_history_consecutive_duplicate_replaces_entry():
    history = add_history_entry([], "same query", "All PDFs", 5, 2)
    # Repeating the exact same query/filter/count consecutively replaces the
    # existing entry rather than appending a duplicate.
    history = add_history_entry(history, "same query", "All PDFs", 5, 3)

    assert len(history) == 1
    assert history[0]["query"] == "same query"
    assert history[0]["num_results"] == 3  # replaced with newest entry


def test_history_non_consecutive_duplicate_appends():
    history = add_history_entry([], "same query", "All PDFs", 5, 2)
    history = add_history_entry(history, "other query", "All PDFs", 5, 1)
    # A non-consecutive repeat creates a new top entry rather than replacing.
    history = add_history_entry(history, "same query", "All PDFs", 5, 4)

    assert len(history) == 3
    assert history[0]["query"] == "same query"
    assert history[0]["num_results"] == 4
    assert history[2]["query"] == "same query"
    assert history[2]["num_results"] == 2


def test_clear_history_returns_empty_list():
    history = add_history_entry([], "query", "All PDFs", 5, 1)
    assert history

    cleared = clear_history()
    assert cleared == []
