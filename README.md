# RAG-based-Q-A-system-
RAG-based Q&amp;A system over course materials using ChromaDB, Sentence-Transformers, Gemini, and Streamlit.
# 📚 Course Q&A RAG System

A Retrieval-Augmented Generation (RAG) system that answers questions about course materials. It retrieves the most relevant passages from three course documents and generates grounded answers with Google Gemini, with source attribution, through an interactive Streamlit interface.

## Features
- End-to-end RAG pipeline built on three course documents
- Semantic search using sentence embeddings
- Answers generated **only from the retrieved context** to reduce hallucinations
- Source display for every answer
- Custom dark-themed Streamlit UI

## Pipeline
1. **Document loading**: three course files are loaded and text is extracted.
2. **Cleaning & chunking**: text is split into 2000 chunks (`[chunk size =2000 , overlap=150`) with metadata.
3. **Embeddings**: each chunk is encoded with `all-MiniLM-L6-v2` ((611,384)dimensions).
1. **Vector store**: embeddings and metadata are stored in a persistent ChromaDB collection.
2. **Retrieval**: the user question is embedded and the top-k similar chunks are fetched.
3. **Generation**: Gemini answers using only the retrieved context.
4. **Deployment**: Streamlit app for interactive Q&A.

## Tech Stack
Python · Sentence-Transformers · ChromaDB · Google Gemini (`google-genai`) · Streamlit · Jupyter

## Project Structure
```
RAG2/
├── code.ipynb        # Pipeline: loading, chunking, embeddings, indexing
├── app.py            # Streamlit application
├── .streamlit/
│   └── config.toml   # Dark theme settings
├── requirements.txt
└── README.md

Set your Gemini API key as an environment variable (never commit it):
```powershell
$env:GEMINI_API_KEY="your_api_key"
```
1. Run `code.ipynb` to build the vector database (`chroma_db/`).
2. Launch the app:
```bash
streamlit run app.py
```
## Author
**Waad Tamer** 


https://github.com/user-attachments/assets/e4fc995a-257d-4d30-b09e-b01f7aaa6b17


