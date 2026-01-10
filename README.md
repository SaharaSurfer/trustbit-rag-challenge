# Enterprise RAG Challenge Solution

This repository contains a Retrieval-Augmented Generation (RAG) system designed as a solution for the [Enterprise RAG Challenge](https://github.com/trustbit/enterprise-rag-challenge). The system extracts information from financial annual reports (PDFs), indexes them using a hybrid search approach, and generates structured answers to specific financial questions.

## Solution Architecture

The pipeline consists of the following stages:

1.  **Data Ingestion & Parsing**:
    *   **Tool**: `marker-pdf`.
    *   **Process**: PDFs are converted to Markdown. Images are currently stripped to focus on textual data.
    *   **Chunking**: Recursive character splitting using `langchain`.

2.  **Indexing & Retrieval**:
    *   **Vector Store**: `ChromaDB`.
    *   **Embeddings**: `BAAI/bge-m3` (Dense retrieval).
    *   **Keyword Search**: BM25 (Sparse retrieval) via `rank_bm25`.
    *   **Hybrid Search**: Candidates are fetched from both Vector and BM25 indices.
    *   **Reranking**: `BAAI/bge-reranker-v2-m3` is used to re-score the top-k candidates.

3.  **Generation & Orchestration**:
    *   **Model**: OpenAI `gpt-4o-mini` via the Structured Outputs API.
    *   **Logic**:
        *   **Router**: Determines if a question is about a single company or a comparison between multiple companies.
        *   **Single Query**: Retrieves relevant chunks and extracts data using Pydantic schemas.
        *   **Comparative Query**: Decomposes the question into sub-questions per company and aggregates results.
    *   **Validation**: The system enforces negative constraints (e.g., returning "N/A" for missing data rather than guessing).

## Limitations

Extracting structured data from complex financial reports presents inherent challenges. The current system is subject to the following technical constraints and model limitations:

1.  **Numerical Scaling Errors**: The LLM occasionally fails to correctly apply unit scaling context (e.g., treating a value listed as €38,366 millions as 38,366.0 instead of 38,366,000,000).
2.  **False Negatives**: The system prompts are engineered to return "N/A" or "False" if the information is not explicitly stated. Consequently, the system may miss valid answers that require minor inference or are located in complex table structures that were split during chunking.
3.  **Overall Instruction Adherence**: The model occasionally overlooks specific system instructions (e.g., negative constraints or formatting rules). This can lead to answers that deviate from the strict schema despite prompt engineering efforts.
4.  **Retrieval Gaps**: Despite the hybrid approach, specific line items in dense financial tables may sometimes be ranked lower than general textual descriptions, leading to missing context for the LLM.
5. **Table Fragmentation:** The current chunking strategy may sever the connection between column headers and row data. Consequently, the LLM might fail to interpret isolated table rows correctly or the retriever might rank them poorly due to missing semantic context.
6.  **Text-Only Analysis**: The pipeline extracts text but currently ignores infographics, charts, and diagrams. Information presented exclusively in visual formats is not retrievable.

## Project Structure

```text
.
├── data/
│   ├── pdfs/                 # Input PDF files
│   ├── processed/            # Intermediate Markdown files
│   ├── chroma_db/            # Vector database
│   ├── subset.csv            # Company name to filename (sha1) mapping metadata
│   └── questions.json        # Input questions
├── src/
│   └── trustbit_rag_challenge/
│       ├── llm/              # OpenAI client, Jinja2 templates
│       ├── scripts/          # Scripts (chunking, indexing, submission creation)
│       ├── config.py         # Configuration settings
│       ├── enums.py          # Enum for question kind
│       ├── logging_utils.py  # Code for setupping logging
│       ├── schemas.py        # Pydantic schemas for LLM, submission and DTOs
│       ├── retriever.py      # Hybrid retrieval implementation
│       └── router.py         # Query routing logic
├── pyproject.toml            # Dependencies
├── .env                      # Environmental variables (OpenAI API key)
└── README.md
```

## Setup & Installation

The project uses **Poetry** for dependency management.

1.  **Prerequisites**
    *   Python 3.11+
    *   Poetry
    *   CUDA-enabled GPU.

2.  **Install Dependencies**
    ```bash
    poetry install
    ```

3.  **Environment Variables**

    Create a `.env` file in the root directory:
    ```bash
    OPENAI_API_KEY=sk-...
    ```

4.  **Data Placement**

    Place the challenge data in the `data/` directory:
    *   PDFs in `data/pdfs/`
    *   `subset.csv` in `data/`
    *   `questions.json` in `data/`

## Running the Pipeline

The pipeline must be run sequentially to prepare the data before generating answers.

### 1. Parse PDFs
Converts PDFs to Markdown. This process is GPU-intensive.
```bash
poetry run process_all_pdfs
```

### 2. Merge & Chunk
Consolidates parsed pages and splits them into chunks for indexing.
```bash
poetry run merge_pdf_chunks
poetry run prepare_rag_chunks
```

### 3. Build Index
Creates the ChromaDB vector store and BM25 indices.
```bash
poetry run index_rag_chunks
```

### 4. Generate Submission
Runs the RAG system on `questions.json` and produces the submission file.
```bash
poetry run generate_submission
```

The final output will be saved in `data/submission_<name>.json`.

### 5. Submitting solution
Checks results against the validation server (if running) and shows current leaderboard:
```bash
poetry run submit
```
