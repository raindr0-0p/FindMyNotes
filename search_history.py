"""Search history helpers for FindMyNotes.

Pure functions operating on plain dictionaries/lists so the history logic can be
tested without Streamlit or the embedding model. History itself lives only in
Streamlit session state.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

MAX_HISTORY_ENTRIES = 10


def make_history_entry(
    query: str,
    source_filter: str,
    result_count: int,
    num_results: int,
) -> dict[str, Any]:
    """Create a single search history entry."""
    return {
        "query": query,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_filter": source_filter,
        "result_count": result_count,
        "num_results": num_results,
    }


def _entry_matches(
    entry: dict[str, Any],
    query: str,
    source_filter: str,
    result_count: int,
) -> bool:
    return (
        entry["query"] == query
        and entry["source_filter"] == source_filter
        and entry["result_count"] == result_count
    )


def add_history_entry(
    history: list[dict[str, Any]],
    query: str,
    source_filter: str,
    result_count: int,
    num_results: int,
    max_entries: int = MAX_HISTORY_ENTRIES,
) -> list[dict[str, Any]]:
    """Add a search to the history, returning a new history list.

    Consecutive duplicates (same query, source filter, and result count as the
    most recent entry) replace the existing entry instead of appending, so a
    repeated search becomes the newest entry without creating an unnecessary
    duplicate. The list is capped at ``max_entries`` most-recent-first entries.
    """
    new_entry = make_history_entry(query, source_filter, result_count, num_results)

    updated = list(history)

    # Avoid a duplicate consecutive entry: replace the most recent entry when it
    # matches the new search exactly.
    if updated and _entry_matches(updated[0], query, source_filter, result_count):
        updated[0] = new_entry
    else:
        updated.insert(0, new_entry)

    # Keep only the most recent entries.
    return updated[:max_entries]


def clear_history() -> list[dict[str, Any]]:
    """Return an empty history list."""
    return []
