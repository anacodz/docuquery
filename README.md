# DocuQuery

A lightweight Retrieval-Augmented Generation (RAG) tool I built to query and extract information from PDF documents. It uses LangChain for orchestration, FAISS for local vector search, and Google's Gemini LLM to generate answers based on the document's context.

## Why I built this
I needed a way to quickly search through large technical PDFs and research papers without manually reading through hundreds of pages. Since I wanted to keep the vector embeddings local and free, I used HuggingFace Sentence Transformers instead of relying on paid embedding APIs.

## Tech Stack
- **Python & Streamlit** (Frontend)
- **LangChain & FAISS** (RAG pipeline and Vector Database)
- **HuggingFace Sentences Transformers** (`all-MiniLM-L6-v2` for local embeddings)
- **Google GenAI** (Gemini 1.5 Pro for response generation)

## Setup Instructions

1. **Clone the repo:**
   ```bash
   git clone https://github.com/your-username/docuquery.git
   cd docuquery
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   # (For Windows: venv\Scripts\activate)
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory and add your Google API key:
   ```env
   GOOGLE_API_KEY="your_api_key_here"
   ```

5. **Run it:**
   ```bash
   streamlit run app.py
   ```

## How it works
1. **Document Parsing:** Extracts raw text from uploaded PDFs using `PyPDF2`.
2. **Chunking & Embedding:** Splits the text into smaller chunks and generates vector embeddings using a local HuggingFace model.
3. **Similarity Search:** When you ask a question, FAISS retrieves the most relevant text chunks based on semantic similarity.
4. **LLM Generation:** The context and question are passed to the Gemini LLM, which generates a natural language answer based *only* on the provided documents.
