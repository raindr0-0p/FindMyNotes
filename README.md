# FindMyNotes

**Try it live: [findmynotes.streamlit.app](https://findmynotes.streamlit.app/)**

FindMyNotes is a semantic search application for PDF documents. Upload one or more PDFs, search by meaning, and find the five most relevant passages with their source filenames, page numbers, and similarity scores.

Built with Python, PyMuPDF, Sentence Transformers, NumPy, and Streamlit.

![FindMyNotes interface](docs/screenshot.png)

## The problem it solves

When you have multiple PDFs, you might remember an idea without remembering the exact wording or which document contains it. Traditional keyword search can miss relevant passages when your query uses different words from the original text.

FindMyNotes uses semantic embeddings to rank passages by similarity to your query. For example, you can ask, *“Why does a Transformer need positional encoding to represent token order?”* and retrieve related passages from your notes even when the wording differs.

## Features

- Upload and search across multiple PDF documents.
- Extract text page by page while preserving 1-based page numbers.
- Split page text into overlapping word chunks without merging content across pages.
- Generate text embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
- Rank passages using cosine similarity calculated with NumPy.
- Display the top five results with passage text, source filename, page number, and similarity score.
- Handle invalid PDFs, PDFs without extractable text, empty queries, and searches before indexing with clear messages.
- Cache the embedding model and retain the search index in the active Streamlit session.

## Live demo

**[Open FindMyNotes](https://findmynotes.streamlit.app/)**

The public demo is hosted on Streamlit Community Cloud. Initial startup can take longer if the application needs to download the pretrained embedding model.

**Privacy note:** In the hosted demo, uploaded PDFs are processed by the cloud-hosted application. Do not upload confidential or sensitive documents. When running the application locally, processing takes place on your own computer after the model has been downloaded.

## Technology stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| PyMuPDF (`pymupdf`) | Page-by-page PDF text extraction |
| Sentence Transformers | Convert document passages and queries into embeddings |
| `all-MiniLM-L6-v2` | Pretrained text embedding model |
| NumPy | Cosine similarity calculations and ranking |
| Streamlit | Web user interface |
| pytest | Automated tests |

## How it works

The application follows this pipeline:

1. **PDF extraction** — `pdf_processor.extract_pages()` extracts text from each page and records its 1-based page number. Pages without extractable text are skipped; scanned/image-only PDFs are not OCR-processed.
2. **Text chunking** — `pdf_processor.chunk_pages()` splits each page into overlapping word chunks. Chunks never cross page boundaries, and each chunk retains its source filename and page number.
3. **Embeddings** — `search_engine.build_index()` converts each chunk into a numerical vector using `all-MiniLM-L6-v2` and stores the vectors alongside the corresponding source metadata.
4. **Semantic retrieval** — `search_engine.search_chunks()` embeds the user's query and compares it with the indexed chunks using cosine similarity.
5. **Ranked results** — The application returns up to five passages ranked by similarity, along with their filenames, page numbers, and scores.

Similarity scores are **not probabilities** or guarantees of relevance. Always check a result against the original passage.

## Project structure

```text
FindMyNotes/
├── app.py                    # Streamlit interface and session workflow
├── pdf_processor.py          # PDF extraction and page-aware chunking
├── search_engine.py          # Embeddings, cosine similarity, and retrieval
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
├── README.md                 # Project documentation
├── docs/
│   └── screenshot.png        # Screenshot of the running application
└── tests/
    ├── test_pdf_processor.py # PDF extraction and chunking unit tests
    ├── test_search_engine.py # Semantic search unit tests
    └── test_real_pipeline.py # Opt-in real-model integration test
```

## Setup on Windows

### 1. Get the project

Clone your GitHub repository or download it, then open a terminal in the project directory. For example:

```bat
cd /d D:\findmynotes
```

### 2. Create and activate a virtual environment

**Command Prompt (CMD):**

```bat
py -m venv .venv
.venv\Scripts\activate.bat
```

**PowerShell:**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

With the virtual environment activated, run:

```bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## The embedding model

FindMyNotes uses the public pretrained model `sentence-transformers/all-MiniLM-L6-v2`. The first model download requires an internet connection. You can download and cache it before launching the application:

```bat
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

Once cached, the model can be reused locally. The hosted demo also needs to download the model in its cloud environment when it is not already cached.

## Run the app

With the virtual environment activated, start FindMyNotes:

```bat
python -m streamlit run app.py --server.fileWatcherType=none
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

The `--server.fileWatcherType=none` option avoids noisy file-watcher import warnings in some Windows environments. You can try the standard command below if the watcher works correctly on your setup:

```bat
python -m streamlit run app.py
```

### Use the application

1. Select one or more PDFs in the uploader.
2. Click **Index documents** and wait for extraction, chunking, and embedding generation to finish.
3. Enter a natural-language query and click **Search**.
4. Review the ranked passages and check the source filename and page number in the original PDF.
5. If you change the selected uploads, index them again before searching so the index reflects the current selection.

## Run the tests

### Regular unit tests

These tests use deterministic test embeddings and do not need to download the real model:

```bat
python -m pytest tests/test_search_engine.py tests/test_pdf_processor.py -v
```

### Opt-in real-model integration test

The integration test exercises the PDF-to-search pipeline with the real embedding model and is skipped by default. Download/cache the model first, then enable the test.

**Command Prompt (CMD):**

```bat
set RUN_REAL_MODEL_INTEGRATION=1
python -m pytest tests/test_real_pipeline.py -v
set RUN_REAL_MODEL_INTEGRATION=
```

**PowerShell:**

```powershell
$env:RUN_REAL_MODEL_INTEGRATION = "1"
python -m pytest tests/test_real_pipeline.py -v
Remove-Item Env:RUN_REAL_MODEL_INTEGRATION
```

To run the entire test suite with the integration test enabled:

```bat
set RUN_REAL_MODEL_INTEGRATION=1
python -m pytest -v
set RUN_REAL_MODEL_INTEGRATION=
```

## Limitations

- PDF files are the only supported input format.
- Scanned or image-only PDFs are not processed with OCR.
- The app retrieves and ranks passages; it does not generate answers with an LLM or act as a chatbot.
- The index is held in Streamlit session state and is not persisted in a database. Restarting the app clears it.
- Semantic similarity is useful for ranking likely matches but does not guarantee that every result is relevant.
- The hosted demo processes uploaded documents in the cloud. Avoid uploading private or sensitive material to it.

## Possible future improvements

These ideas are intentionally outside the current MVP scope: retrieval evaluation on a labeled query set, hybrid keyword-and-semantic search, persistent indexing, support for more document formats, and OCR for scanned documents.
