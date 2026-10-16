\# FindMyNotes — Project Instructions



\## Project Goal



Build a small, local-first semantic search application for PDF documents in Python. Users should be able to upload multiple PDFs, search their contents by meaning, and receive the five most relevant passages with correct filenames and page numbers.



\## Frozen MVP Requirements



1\. Upload multiple PDF documents.

2\. Extract text page by page using PyMuPDF.

3\. Split extracted text into overlapping chunks, preserving source filename and page number.

4\. Generate embeddings using `sentence-transformers/all-MiniLM-L6-v2`.

5\. Use NumPy to calculate cosine similarity and rank document chunks.

6\. Display the top five matching passages in a Streamlit interface.

7\. Show each passage's source filename, page number, and similarity score.

8\. Handle invalid PDFs, PDFs without extractable text, empty queries, and missing documents gracefully.

9\. Include automated tests and a README with setup and run instructions.



\## Technology Stack



\- Python

\- PyMuPDF

\- Sentence Transformers

\- NumPy

\- Streamlit

\- pytest

\- Git



\## Architecture



\- `pdf\_processor.py`: PDF extraction and text chunking.

\- `search\_engine.py`: Embeddings, cosine similarity, and retrieval.

\- `app.py`: Streamlit user interface.

\- `tests/`: Automated tests.

\- `requirements.txt`: Python dependencies.



Keep the architecture simple. In-memory storage is sufficient for the MVP.



\## Strict Scope Rules



Do not add features until the existing MVP passes its acceptance tests.



Out of scope:

\- LLM-generated answers or chatbots

\- OCR for scanned PDFs

\- Word documents or additional document formats

\- Databases or external vector databases

\- Authentication or multi-user support

\- Cloud deployment

\- Agents, elaborate frameworks, or unnecessary abstractions



Do not change the technology stack or introduce dependencies without a clear technical need and explicit approval.



\## Development Rules



1\. Work on one explicitly requested milestone at a time.

2\. Inspect existing code before modifying it.

3\. Prefer small, understandable changes over broad rewrites.

4\. Write tests for relevant functionality and run them after changes.

5\. Never claim tests passed unless they were actually executed successfully.

6\. Preserve source filenames and page metadata throughout the retrieval pipeline.

7\. Handle expected errors clearly; do not hide errors with broad exception handling.

8\. Explain significant implementation decisions and any limitations.

9\. Do not overwrite working code or unrelated files unnecessarily.

10\. If a requirement is ambiguous, explain the uncertainty rather than silently expanding scope.



\## Acceptance Criteria



The MVP is complete only when:

\- Multiple PDFs can be indexed and searched.

\- Semantic queries retrieve useful passages in manual test cases.

\- Results are ranked by cosine similarity.

\- Filenames and page numbers are correct.

\- Invalid and text-empty PDFs are handled gracefully.

\- Empty queries and searches without indexed documents are handled.

\- Automated tests pass.

\- README instructions allow a new user to run the application.



Until these criteria are met, prioritize correctness, debugging, and testing over new features.

