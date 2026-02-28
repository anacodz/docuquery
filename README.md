# Enterprise RAG Document AI Assistant

An advanced Retrieval-Augmented Generation (RAG) web application that enables users to query complex PDF documents. Built using Python, Streamlit, Langchain, and Google GenAI. This project demonstrates core Data/ML Engineering concepts including vector embeddings, similarity search, and generative AI orchestration.

## Features
- **Document Ingestion:** Securely parse and extract text from uploaded PDF documents.
- **Embedded Vector Database:** Leverages FAISS for fast similarity search using HuggingFace sentence transformer embeddings.
- **Generative AI Responses:** Integrate Google's Gemini models for accurate and context-aware natural language responses based on the PDF content.
- **Responsive UI:** Streamlit frontend for a seamless and intuitive user experience.

## Tech Stack
- **Frontend:** Streamlit
- **Backend Infrastructure:** Python, LangChain, FAISS
- **AI Models:** Google Gemini (LLM) & HuggingFace (Embeddings)
- **Deployment Strategy:** AWS EC2 / Render / Local deployment

## Architecture Overview
1. **User Uploads PDF:** The file is ingested and the text is parsed into chunks using a character-level text splitter.
2. **Generating Embeddings:** Each text chunk is converted into semantic vector embeddings using open-source models (HuggingFace `all-MiniLM-L6-v2`).
3. **Similarity Search (FAISS):** When a user asks a query, the query is also converted to an embedding. FAISS quickly finds the top-K most similar text chunks.
4. **LLM Generation:** The context (retrieved chunks) and the user's query are passed to the Gemini LLM to synthesize a precise answer.

## Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/rag-document-ai.git
   cd rag-document-ai
   ```

2. **Create a virtual environment & Install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate # For Mac/Linux
   # venv\Scripts\activate # For Windows
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables:**
   - Copy the `.env.example` file to `.env`
   - Add your Google API key to `.env` (Get it from [Google AI Studio](https://aistudio.google.com/))
   ```bash
   cp .env.example .env
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## Future Enhancements
- Dockerizing the application for automated deployments.
- Deploying the Streamlit frontend to AWS or Streamlit Cloud.
- Expanding support for .docx, .txt, and image-based PDFs (OCR).
