<img width="1680" height="1050" alt="Screenshot 2025-07-17 at 5 17 03 PM" src="https://github.com/user-attachments/assets/752393af-98ad-4f7b-8170-4b2d3ab3945d" />

<img width="1680" height="1050" alt="Screenshot 2025-07-17 at 5 16 05 PM" src="https://github.com/user-attachments/assets/03d8fb4f-232a-479c-8b42-bac300ba8208" />




# RAG-QA-System

This project implements a Retrieval-Augmented Generation (RAG) system for document-based medical question answering using open-source LLMs. It is designed to be modular, clean, and suitable for both Jupyter Notebook environments and deployment with Streamlit.

## System Requirements and Scenario

**Input:** Provided clinical or structured medical documents (e.g., PDF, TXT).

**Objective:** An end-to-end RAG pipeline that extracts, chunks, embeds, stores, retrieves, and answers questions from the content using an LLM.

## Features

-   **Document Ingestion & Chunking:** Loads and extracts text from PDF/TXT files, chunks text into 200-500 token passages with NLTK for intelligent boundary detection and overlap.
-   **Embedding & Vector Index:** Generates dense vector embeddings using `all-MiniLM-L6-v2` (SentenceTransformers) and stores them in a FAISS index.
-   **Semantic Search:** Retrieves top-k most relevant chunks using cosine similarity.
-   **Open-Source LLM Integration:** Utilizes `mistralai/Mistral-7B-Instruct-v0.2` (or similar) from HuggingFace for question answering.
-   **RAG QA Pipeline:** A complete workflow from question input to formatted answer.
-   **Advanced Features:**
    -   **Streamlit Web Interface:** Allows file upload, question input, answer display, and optional display of retrieved context chunks.
    -   **In-memory Caching:** For repeat user queries (answer + contexts).
    -   **Maximal Marginal Relevance (MMR) Reranking:** For diversity among top-k chunks.

## Project Structure

```
RAG-Project/
├───RAG-Proj/
│   ├───__init__.py
│   ├───.env
│   ├───app.py                 # Streamlit application
│   ├───requirements.txt
│   ├───documents/             # Sample documents
│   │   └───9241544228_eng.pdf
│   └───src/
│       ├───__init__.py
│       ├───document_processing.py # Handles document loading and chunking
│       ├───vector_store.py        # Manages FAISS index creation
│       ├───retrieval.py           # Implements semantic search and MMR reranking
│       ├───llm_integration.py     # Handles LLM interaction via HuggingFace Inference API
│       └───qa_pipeline.py         # Orchestrates the RAG workflow
├───temp_docs/                 # Temporary directory for uploaded documents
└───main.py                    # Example script for local execution
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/saquibkhan2/RAG-QA-System.git
    cd RAG-QA-System
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r RAG-Proj/requirements.txt
    ```

4.  **Set up HuggingFace API Token:**
    Create a `.env` file in the `RAG-Proj/` directory and add your HuggingFace API token:
    ```
    HF_TOKEN="your_huggingface_api_token_here"
    ```
    You can obtain a token from [HuggingFace Settings](https://huggingface.co/settings/tokens).

## How to Run

### 1. Streamlit Web Interface

To run the Streamlit application:

```bash
streamlit run RAG-Proj/app.py
```

This will open the web interface in your browser, where you can upload documents, ask questions, and see the answers.

### 2. Local Execution (Command Line)

To run the example queries from the command line:

```bash
python main.py
```

This script will load the sample PDF, process it, and answer two predefined medical questions, demonstrating the core RAG pipeline.

## Design and Implementation Decisions

-   **Modularity:** The code is organized into distinct modules (`document_processing`, `vector_store`, `retrieval`, `llm_integration`, `qa_pipeline`) to enhance maintainability and readability.
-   **NLTK for Chunking:** NLTK's `sent_tokenize` and `word_tokenize` are used for more intelligent text splitting, preserving sentence boundaries and word integrity.
-   **FAISS for Vector Index:** FAISS provides efficient similarity search for fast retrieval of relevant document chunks.
-   **SentenceTransformers for Embeddings:** `all-MiniLM-L6-v2` is chosen for its balance of performance and efficiency in generating dense vector embeddings.
-   **HuggingFace Inference API:** The LLM (`mistralai/Mistral-7B-Instruct-v0.2`) is accessed remotely via the HuggingFace Inference API, avoiding local resource intensive LLM loading.
-   **MMR Reranking:** Implemented to ensure diversity among the retrieved context chunks, preventing redundancy and improving the quality of context provided to the LLM.
-   **In-memory Caching:** A simple dictionary-based cache is used to store answers for repeated queries, improving response times for common questions.

## Known Limitations

-   **Resource Requirements:** While the LLM is remote, embedding generation and FAISS indexing can still be memory-intensive for very large documents.
-   **NLTK Download:** The initial run might require downloading NLTK data, which needs an internet connection.
-   **Cache Scope:** The in-memory cache is cleared when the Streamlit application restarts or the `RAGPipeline` instance is reloaded.
-   **Error Handling:** Basic error handling is in place, but more robust error management could be added for production environments.
-   **LLM Response Variability:** Answers depend on the LLM's capabilities and the quality of retrieved context.

## How AI Assistants Were Used

This project was developed with the assistance of a Gemini AI model. The AI assistant helped in:
-   Refactoring the initial monolithic `rag_pipeline.py` into modular components.
-   Generating docstrings and inline comments for improved code clarity.
-   Implementing the MMR reranking algorithm.
-   Structuring the `README.md` and providing explanations for design choices.
-   Debugging and resolving issues related to Git operations and file paths.
