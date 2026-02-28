# DocuQuery ✨

A stylish, enterprise-grade Retrieval-Augmented Generation (RAG) tool I built to query, cross-reference, and extract information from multiple PDF documents simultaneously. It uses LangChain for orchestration, FAISS for local vector search, and Google's newest Gemini models to generate highly accurate answers based strictly on the document context.

## 🚀 Key Features
- **100% Multilingual Support**: Capable of ingesting, understanding, and citing documents written in over 50+ languages simultaneously (Hindi, Spanish, French, etc.)
- **Multi-Document Synthesis**: Upload multiple PDFs at once. The AI maintains source-truth metadata and can cross-reference differences between documents.
- **Dynamic UI Theming**: Built-in sleek color palettes (Elegant Pink, Ocean Blue, Midnight Dark, Forest Green) that dynamically adjust the entire app's CSS.
- **Local Embeddings**: Keeps processing cost-free and secure by generating vectors locally before querying the LLM.
- **Graceful Error Handling**: Skips corrupted files non-destructively instead of crashing the pipeline.

## 🛠️ Tech Stack
- **Python & Streamlit** (Frontend & Dynamic Theming)
- **LangChain & FAISS** (RAG orchestration and high-speed Vector Database)
- **HuggingFace Sentences Transformers** (`paraphrase-multilingual-MiniLM-L12-v2` for cross-lingual embeddings)
- **Google GenAI** (`gemini-flash-latest` for low-latency reasoning and response generation)

## 💻 Setup Instructions

1. **Clone the repo:**
   ```bash
   git clone https://github.com/anacodz/docuquery.git
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

5. **Run it Locally:**
   ```bash
   streamlit run app.py
   ```

## 🧠 How the Architecture works
1. **Document Parsing & Metadata:** Extracts raw text from uploaded PDFs using `PyPDF2` and tags each chunk with its source filename.
2. **Chunking & local Embedding:** Splits the text into small overlapped chunks and maps them into semantic vectors using HuggingFace.
3. **Similarity Search:** When a user asks a complex question, FAISS retrieves the top 12 most mathematically relevant text chunks across all documents.
4. **LLM Synthesis:** The deep context and question are passed to the Gemini LLM, which synthesizes a natural language answer and explicitly cites its source files.
