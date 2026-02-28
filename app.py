import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# load environment variables
load_dotenv()

def extract_text_from_pdfs(pdf_list):
    text = ""
    for pdf in pdf_list:
        reader = PdfReader(pdf)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text

def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def create_vector_db(chunks):
    # using local embeddings to save costs
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_texts(chunks, embedding=embeddings)
    db.save_local("faiss_index_store")

def setup_qa_chain():
    template = """
    Answer the user's question based on the provided document context.
    If the answer isn't in the context, just say "I couldn't find this in the uploaded document." Don't make things up.

    Context:
    {context}
    
    Question: 
    {question}

    Answer:
    """
    
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.2)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)

def process_query(query):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    try:
        db = FAISS.load_local("faiss_index_store", embeddings, allow_dangerous_deserialization=True)
        relevant_docs = db.similarity_search(query, k=4)

        chain = setup_qa_chain()
        with st.spinner("🧠 Synthesizing answer..."):
            res = chain.invoke(
                {"input_documents": relevant_docs, "question": query}
            )
        
        st.markdown("<div class='success-message'>✨ <strong>Answer Generated!</strong></div>", unsafe_allow_html=True)
        st.info(res["output_text"])
        
        with st.expander("📚 View Extracted Source Context"):
            for i, doc in enumerate(relevant_docs):
                st.markdown(f"**Source {i+1}**:")
                st.markdown(f"> *{doc.page_content}*")
                st.divider()
    except Exception as e:
        st.error(f"Something went wrong. Is the API key configured properly? Error: {str(e)}")

def main():
    st.set_page_config(page_title="DocuQuery | AI PDF Assistant", page_icon="⚡", layout="wide")
    
    # Custom CSS for a beautiful, colorful UI
    st.markdown("""
    <style>
        .main {
            background-color: #f8fafc;
        }
        h1 {
            color: #1e293b;
            font-family: 'Inter', sans-serif;
            font-weight: 800;
        }
        .subtitle {
            font-size: 1.2rem;
            color: #475569;
            margin-bottom: 2rem;
        }
        .stButton>button {
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
        }
        .success-message {
            background-color: #dcfce7;
            color: #166534;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            border-left: 4px solid #22c55e;
        }
        .css-1d391kg {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
    </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2916/2916315.png", width=100)
    with col2:
        st.title("DocuQuery ⚡")
        st.markdown("<p class='subtitle'>The Lightning-Fast <strong>RAG Engine</strong> for your Enterprise PDFs.</p>", unsafe_allow_html=True)

    api_key = os.getenv("GOOGLE_API_KEY")

    with st.sidebar:
        st.markdown("## 📁 Data Pipeline")
        st.markdown("Upload files to generate embeddings.")
        uploaded_pdfs = st.file_uploader("Drop PDFs here", accept_multiple_files=True)
        
        if st.button("🚀 Process & Embed Documents"):
            if not api_key:
                st.error("🔑 Missing Google API Key in settings.")
            elif not uploaded_pdfs:
                st.warning("📄 Please upload a file first.")
            else:
                with st.spinner("Extracting text & building FAISS Vector Database..."):
                    raw_text = extract_text_from_pdfs(uploaded_pdfs)
                    chunks = split_text_into_chunks(raw_text)
                    if not chunks:
                        st.error("No readable text found in the PDF.")
                    else:
                        create_vector_db(chunks)
                        st.success("✅ Database built! Ready for queries.")

    st.markdown("---")
    
    if os.path.exists("faiss_index_store"):
        st.markdown("### 🔍 Semantic Search")
        query = st.chat_input("Ask a complex question about your documents...")
        if query:
            st.markdown(f"**You asked:** *{query}*")
            process_query(query)
    else:
        st.info("👈 Please start the pipeline by uploading a document in the sidebar.")

if __name__ == "__main__":
    main()
