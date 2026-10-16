"""FindMyNotes — Streamlit interface for semantic PDF search."""

from __future__ import annotations

import streamlit as st

from pdf_processor import chunk_pages, extract_pages
from search_engine import build_index, search_chunks

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


@st.cache_resource(show_spinner="Loading embedding model ...")
def load_embedding_model():
    """Load the embedding model once per Streamlit process.

    ``local_files_only=True`` ensures the model is never downloaded; it must
    already be present in the local cache.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def main() -> None:
    st.set_page_config(page_title="FindMyNotes")
    st.title("FindMyNotes")
    st.write(
        "FindMyNotes lets you search across multiple PDF documents by meaning. "
        "Upload one or more PDFs, index them, then enter a query to find the most "
        "relevant passages, each with its source filename, page number, and "
        "similarity score."
    )

    uploaded_files = st.file_uploader(
        "Upload PDF documents", type=["pdf"], accept_multiple_files=True
    )

    # Load the embedding model (cached once per process).
    try:
        model = load_embedding_model()
    except Exception as exc:  # shown cleanly to the user, no traceback
        st.error(
            "Could not load the embedding model. Make sure "
            f"'{MODEL_NAME}' is downloaded and cached locally. Details: {exc}"
        )
        st.stop()

    # ---------------- Indexing ----------------
    st.subheader("Index documents")
    if st.button("Index documents"):
        if not uploaded_files:
            st.error("Please select at least one PDF to index.")
        else:
            all_chunks: list[dict] = []
            indexed_names: list[str] = []
            failures: list[tuple[str, str]] = []

            for uploaded_file in uploaded_files:
                try:
                    pdf_bytes = uploaded_file.read()
                    pages = extract_pages(pdf_bytes)
                    chunks = chunk_pages(pages, uploaded_file.name)
                    if chunks:
                        all_chunks.extend(chunks)
                        indexed_names.append(uploaded_file.name)
                    else:
                        failures.append(
                            (uploaded_file.name, "No extractable text found.")
                        )
                except Exception as exc:
                    failures.append((uploaded_file.name, f"Could not process: {exc}"))

            if not all_chunks:
                st.error(
                    "No text chunks were produced from the uploaded PDFs. "
                    "Make sure the PDFs contain extractable text (scanned "
                    "image-only pages are not supported)."
                )
                # Drop any stale index so it cannot be searched.
                st.session_state.pop("index", None)
                st.session_state.pop("indexed_files", None)
                st.session_state.pop("indexed_chunks", None)
            else:
                index = build_index(all_chunks, model)
                st.session_state["index"] = index
                st.session_state["indexed_files"] = indexed_names
                st.session_state["indexed_chunks"] = len(all_chunks)
                st.session_state["index_failures"] = failures
                st.success(
                    f"Indexed {len(indexed_names)} document(s) and "
                    f"{len(all_chunks)} text chunk(s)."
                )
                if failures:
                    detail = "; ".join(
                        f"{name} ({reason})" for name, reason in failures
                    )
                    st.warning(f"Some documents could not be indexed: {detail}")

    # Warn when the upload selection changed after indexing, so the user does
    # not search a stale index while believing it reflects the new selection.
    current_names = [f.name for f in uploaded_files] if uploaded_files else []
    indexed_names_state = st.session_state.get("indexed_files")
    if indexed_names_state and set(current_names) != set(indexed_names_state):
        st.warning(
            "The selected PDFs have changed since the last indexing. "
            "Click 'Index documents' to refresh the index before searching; "
            "searching now would use the previous index."
        )

    # ---------------- Search ----------------
    st.subheader("Search")
    index = st.session_state.get("index")
    query = st.text_input("Search query", placeholder="Enter your search query...")

    if st.button("Search"):
        if index is None:
            st.error(
                "No documents have been indexed yet. Click 'Index documents' first."
            )
        elif not query or not query.strip():
            st.error("Please enter a non-empty search query.")
        else:
            try:
                results = search_chunks(query.strip(), index, model, top_k=TOP_K)
                if not results:
                    st.info("No results found.")
                else:
                    st.write(f"Top {len(results)} results:")
                    for rank, result in enumerate(results, start=1):
                        st.write(
                            f"**{rank}.** {result['filename']} — page {result['page']} "
                            f"(similarity: {result['similarity']:.4f})"
                        )
                        # Passage text as plain text, not Markdown/HTML.
                        st.text(result["text"])
            except ValueError as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
