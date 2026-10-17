"""FindMyNotes — Streamlit interface for semantic PDF search.

``app.py`` owns behaviour and session state. All visual design (tokens,
stylesheet, HTML builders) lives in ``findmynotes_ui.py``.
"""

from __future__ import annotations

import streamlit as st

import findmynotes_ui as ui
from pdf_processor import chunk_pages, extract_pages
from search_engine import build_index, generate_search_results_csv, search_chunks
from search_history import MAX_HISTORY_ENTRIES, add_history_entry, clear_history

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

CSV_FILENAME = "findmynotes_results.csv"
HISTORY_SELECT_PLACEHOLDER = "Select a past search..."
ALL_PDFS = "All PDFs"

THEME_KEY = "fm_dark_mode"  # True = dark. Lives in session state, so reruns keep it.
THEME_PARAM = "theme"  # ?theme=dark|light keeps the choice across a browser refresh.


@st.cache_resource(show_spinner="Loading embedding model ...")
def load_embedding_model():
    """Load the embedding model once per Streamlit process.

    ``local_files_only=True`` ensures the model is never downloaded; it must
    already be present in the local cache.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


# --------------------------------------------------------------------------- #
# Session-state helpers
# --------------------------------------------------------------------------- #


def _clear_current_search() -> None:
    """Drop the current results so stale results can never be shown or exported."""
    for key in ("search_results", "search_query", "search_meta", "search_empty"):
        st.session_state.pop(key, None)


def _flash(kind: str, message: str) -> None:
    """Queue a one-time message to show on the next run (survives st.rerun)."""
    st.session_state.setdefault("fm_flash", []).append((kind, message))


def _show_flash() -> None:
    for kind, message in st.session_state.pop("fm_flash", []):
        getattr(st, kind)(message)


# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #


def _initial_dark() -> bool:
    try:
        return st.query_params.get(THEME_PARAM) == "dark"
    except Exception:  # query params unavailable on very old Streamlit
        return False


def _persist_theme() -> None:
    try:
        st.query_params[THEME_PARAM] = "dark" if st.session_state[THEME_KEY] else "light"
    except Exception:
        pass


def _render_header() -> None:
    with ui.compat(st.container, key="fm_header"):
        brand, toggle = ui.compat(st.columns, [6, 1.4], vertical_alignment="center")
        with brand:
            ui.html_block(ui.brand_html(dark=bool(st.session_state[THEME_KEY])))
        with toggle:
            st.toggle("Dark mode", key=THEME_KEY, on_change=_persist_theme)


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #


def _index_documents(widget_files, model, alerts) -> None:
    """Extract, chunk and index the uploaded PDFs (logic unchanged)."""
    if not widget_files:
        with alerts:
            st.error("Please select at least one PDF to index.")
        return

    all_chunks: list[dict] = []
    indexed_names: list[str] = []
    failures: list[tuple[str, str]] = []

    with st.spinner("Reading and indexing your PDFs…"):
        for uploaded_file in widget_files:
            try:
                pdf_bytes = uploaded_file.read()
                pages = extract_pages(pdf_bytes)
                chunks = chunk_pages(pages, uploaded_file.name)
                if chunks:
                    all_chunks.extend(chunks)
                    indexed_names.append(uploaded_file.name)
                else:
                    failures.append((uploaded_file.name, "No extractable text found."))
            except Exception as exc:
                failures.append((uploaded_file.name, f"Could not process: {exc}"))

        if all_chunks:
            index = build_index(all_chunks, model)

    # The index is changing, so the current results no longer describe it.
    _clear_current_search()
    st.session_state["index_failures"] = failures

    if not all_chunks:
        _flash(
            "error",
            "No text chunks were produced from the uploaded PDFs. "
            "Make sure the PDFs contain extractable text (scanned "
            "image-only pages are not supported).",
        )
        # Drop any stale index so it cannot be searched.
        st.session_state.pop("index", None)
        st.session_state.pop("indexed_files", None)
        st.session_state.pop("indexed_chunks", None)
    else:
        st.session_state["index"] = index
        st.session_state["indexed_files"] = indexed_names
        st.session_state["indexed_chunks"] = len(all_chunks)
        _flash(
            "success",
            f"Indexed {len(indexed_names)} document(s) and "
            f"{len(all_chunks)} text chunk(s).",
        )
    # Redraw so the Documents section settles into its final state at once.
    st.rerun()


def _render_documents(model) -> None:
    index = st.session_state.get("index")
    indexed_files = st.session_state.get("indexed_files")
    indexed_chunks = st.session_state.get("indexed_chunks")

    with ui.compat(st.container, key="fm_docs"):
        summary_slot = st.container()
        alerts_slot = st.container()

        label = "Add documents" if index is None else "Add or replace documents"
        with ui.compat(st.expander, label, expanded=index is None):
            ui.html_block(ui.upload_help_html())
            st.file_uploader(
                "Upload PDF documents",
                type=["pdf"],
                accept_multiple_files=True,
                key="index_uploader",
                label_visibility="collapsed",
            )
            index_clicked = st.button("Index documents", key="index_button", type="primary")

        # Re-read uploaded files from the widget for indexing.
        widget_files = st.session_state.get("index_uploader")

        if index_clicked:
            _index_documents(widget_files, model, alerts_slot)  # reruns on completion

        with alerts_slot:
            _show_flash()
            failures = st.session_state.get("index_failures") or []
            if failures:
                detail = "; ".join(f"{name} ({reason})" for name, reason in failures)
                st.warning(f"Some documents could not be indexed: {detail}")

            # Warn when the upload selection changed after indexing.
            current_names = [f.name for f in widget_files] if widget_files else []
            if indexed_files and set(current_names) != set(indexed_files):
                st.info(
                    "The selected PDFs have changed since the last indexing. "
                    "Click 'Index documents' to refresh the index before searching."
                )

        with summary_slot:
            ui.html_block(ui.documents_head_html(indexed_files, indexed_chunks))


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #


def _render_search_panel(index, indexed_files):
    """Draw the search form. Returns (query, source_filename, result_count, clicked)."""
    # Pre-populate the query input when a user loads a past search from history.
    # Using a session-state key so the loaded query actually appears in the input
    # after st.rerun() (the value= argument only sets the initial widget value).
    if "history_query" in st.session_state:
        st.session_state["search_query_input"] = st.session_state.pop("history_query")

    with ui.compat(st.container, key="fm_search_panel"):
        with ui.compat(st.form, "fm_search_form", border=False):
            input_col, button_col = ui.compat(st.columns, [5, 1.15], vertical_alignment="center")
            with input_col:
                query = st.text_input(
                    "Search query",
                    placeholder="Describe what you’re looking for",
                    key="search_query_input",
                    label_visibility="collapsed",
                )
            with button_col:
                search_clicked = ui.compat(
                    st.form_submit_button,
                    "Search",
                    key="search_button",
                    type="primary" if index is not None else "secondary",
                )

            source_col, count_col = st.columns([3, 1])
            with source_col:
                source_choice = st.selectbox(
                    "Source PDF",
                    [ALL_PDFS] + indexed_files,
                    index=0,
                    key="source_filter",
                )
            with count_col:
                result_count = st.number_input(
                    "Results",
                    min_value=1,
                    max_value=100,
                    value=5,
                    step=1,
                    key="result_count",
                )

    source_filename = None if source_choice == ALL_PDFS else source_choice
    return query, source_filename, result_count, search_clicked


def _run_search(model, index, query, source_filename, result_count) -> None:
    """Validate and run a search (logic unchanged); results go to session state."""
    if index is None:
        st.error("No documents have been indexed yet. Click 'Index documents' first.")
        return
    if not query or not query.strip():
        st.error("Please enter a non-empty search query.")
        # Clear any stale results so a failed search cannot export them.
        _clear_current_search()
        return

    try:
        with st.spinner("Searching…"):
            results = search_chunks(
                query.strip(),
                index,
                model,
                top_k=result_count,
                source_filename=source_filename,
            )
        scope = source_filename if source_filename is not None else ALL_PDFS
        # A new search replaces any prior current results so stale results are
        # never exported as if they belonged to a new search.
        _clear_current_search()
        if not results:
            st.session_state["search_empty"] = {"query": query.strip(), "source": scope}
        else:
            # Stored for the results list and the CSV download button, which
            # persist across Streamlit reruns.
            st.session_state["search_results"] = results
            st.session_state["search_query"] = query.strip()
            st.session_state["search_meta"] = {"source": scope, "requested": result_count}

        # Record every successfully completed search, including zero-result
        # searches.
        st.session_state["search_history"] = add_history_entry(
            st.session_state["search_history"],
            query.strip(),
            scope,
            result_count,
            len(results),
        )
    except ValueError as exc:
        st.error(str(exc))


def _render_history_section() -> None:
    """Quiet, collapsed row directly under the search panel."""
    history = st.session_state.get("search_history", [])

    with ui.compat(st.container, key="fm_history"):
        with st.expander("Recent searches", expanded=False):
            if not history:
                ui.html_block(ui.history_empty_html(MAX_HISTORY_ENTRIES))
                return

            history_options = [HISTORY_SELECT_PLACEHOLDER] + [
                f"{entry['query']} ({entry['source_filter']})" for entry in history
            ]
            selected_history = st.selectbox(
                "Load a past search",
                history_options,
                index=0,
                key="history_select",
                label_visibility="collapsed",
            )
            col_rerun, col_clear = st.columns([1, 1])
            with col_rerun:
                load_clicked = st.button("Load selected query", key="history_load")
            with col_clear:
                clear_clicked = st.button(
                    "Clear search history", key="history_clear", type="tertiary"
                )

            if load_clicked and selected_history != HISTORY_SELECT_PLACEHOLDER:
                chosen_index = history_options.index(selected_history) - 1
                chosen_entry = history[chosen_index]
                st.session_state["history_query"] = chosen_entry["query"]
                st.rerun()

            if clear_clicked:
                st.session_state["search_history"] = clear_history()
                st.rerun()


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


def _render_results(index) -> None:
    """Draw the current results from session state (so any rerun keeps them)."""
    results = st.session_state.get("search_results")
    if results:
        query = st.session_state.get("search_query", "")
        meta = st.session_state.get("search_meta") or {}
        with ui.compat(st.container, key="fm_results_head"):
            head_col, csv_col = ui.compat(st.columns, [3, 1.35], vertical_alignment="center")
            with head_col:
                ui.html_block(ui.results_head_html(len(results), query, meta.get("source")))
            with csv_col:
                csv_data = generate_search_results_csv(results, query)
                st.download_button(
                    label="Download results as CSV",
                    data=csv_data,
                    file_name=CSV_FILENAME,
                    mime="text/csv",
                    key="download_csv",
                )
        ui.html_block(ui.score_note_html())
        ui.html_block(ui.results_html(results, query))
    elif st.session_state.get("search_empty"):
        st.info("No results found.")
        ui.html_block('<p class="fm-quiet">Try different wording, or search across all PDFs.</p>')
    else:
        ui.html_block(ui.empty_state_html("ready" if index is not None else "no_documents"))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def _configure_page() -> None:
    options = dict(
        page_title="FindMyNotes",
        layout="centered",
        initial_sidebar_state="collapsed",
        menu_items={"Get help": None, "Report a bug": None, "About": None},
    )
    try:
        st.set_page_config(page_icon=ui.page_icon(), **options)
    except Exception:  # an unsupported icon format must never stop the app
        st.set_page_config(**options)


def main() -> None:
    _configure_page()

    if "search_history" not in st.session_state:
        st.session_state["search_history"] = []
    if THEME_KEY not in st.session_state:
        st.session_state[THEME_KEY] = _initial_dark()

    ui.inject_theme(dark=bool(st.session_state[THEME_KEY]))
    _render_header()

    # Load the embedding model (cached once per process).
    try:
        model = load_embedding_model()
    except Exception as exc:  # shown cleanly to the user, no traceback
        st.error(
            "Could not load the embedding model. Make sure "
            f"'{MODEL_NAME}' is downloaded and cached locally. Details: {exc}"
        )
        st.stop()

    _render_documents(model)

    index = st.session_state.get("index")
    indexed_files = st.session_state.get("indexed_files") or []

    with ui.compat(st.container, key="fm_search"):
        query, source_filename, result_count, search_clicked = _render_search_panel(
            index, indexed_files
        )
        feedback = st.container()
        if search_clicked:
            with feedback:
                _run_search(model, index, query, source_filename, result_count)
        _render_history_section()

    _render_results(index)


if __name__ == "__main__":
    main()
