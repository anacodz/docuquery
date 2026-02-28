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

def extract_text_and_metadatas_from_pdfs(pdf_list):
    text_chunks = []
    metadatas = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    for pdf in pdf_list:
        pdf_text = ""
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    pdf_text += extracted + "\n"
        except Exception as e:
            st.warning(f"⚠️ Skipping '{pdf.name}' because it could not be read. Please make sure it is a valid PDF file.")
            continue
                
        # Split text for this specific document
        if pdf_text.strip():
            chunks = splitter.split_text(pdf_text)
            text_chunks.extend(chunks)
            # Tag each chunk with the name of the file it came from
            metadatas.extend([{"source": pdf.name}] * len(chunks))
        
    return text_chunks, metadatas

def create_vector_db(text_chunks, metadatas):
    # using local embeddings to save costs
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Store the actual file names alongside the vectors
    db = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadatas)
    db.save_local("faiss_index_store")

def setup_qa_chain():
    template = """
    You are an intelligent AI assistant analyzing multiple documents.
    Answer the user's question based on the provided document context below. 
    The context contains snippets from different source documents. If the user asks you to compare documents, clearly state the differences based on the provided text.
    If the answer isn't in the context, just say "I couldn't find this in the uploaded documents." Don't make things up.

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
        # Fetch more context (12 chunks) so the AI has enough text to compare multiple documents
        relevant_docs = db.similarity_search(query, k=12)

        chain = setup_qa_chain()
        with st.spinner("🧠 Synthesizing answer from multiple sources..."):
            res = chain.invoke(
                {"input_documents": relevant_docs, "question": query}
            )
        
        st.markdown("<div class='success-message'>✨ <strong>Answer Generated!</strong></div>", unsafe_allow_html=True)
        st.info(res["output_text"])
        
        with st.expander("📚 View Extracted Source Context"):
            for i, doc in enumerate(relevant_docs[:8]): # Only show top 8 in UI to avoid clutter
                source_file = doc.metadata.get('source', 'Unknown Document')
                st.markdown(f"**Source: {source_file}**")
                st.markdown(f"> *{doc.page_content}*")
                st.divider()
    except Exception as e:
        st.error(f"Something went wrong. Is the API key configured properly? Error: {str(e)}")

def main():
    st.set_page_config(page_title="DocuQuery | AI PDF Assistant", page_icon="🎀", layout="wide")
    
    # Custom CSS for an elegant, chic, and dashing UI
    st.markdown("""
    <style>
        /* Global Background and Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');
        
        .main {
            background: linear-gradient(135deg, #fff0f5 0%, #f3e8ff 100%);
            font-family: 'Poppins', sans-serif;
        }
        
        /* Typography */
        h1, h2, h3 {
            color: #4a044e;
            font-family: 'Poppins', sans-serif;
            font-weight: 800;
            letter-spacing: -0.5px;
        }
        
        .subtitle {
            font-size: 1.3rem;
            color: #701a75;
            font-weight: 400;
            margin-bottom: 2rem;
        }
        
        /* Button Styling */
        .stButton>button {
            background: linear-gradient(45deg, #ec4899 0%, #8b5cf6 100%);
            color: white !important;
            border-radius: 30px;
            border: none;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            font-size: 1.1rem;
            letter-spacing: 0.5px;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            box-shadow: 0 8px 15px rgba(236, 72, 153, 0.3);
            width: 100%;
        }
        .stButton>button:hover {
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 12px 20px rgba(139, 92, 246, 0.4);
        }
        
        /* Success Message Container */
        .success-message {
            background: rgba(253, 232, 243, 0.85);
            backdrop-filter: blur(10px);
            color: #be185d;
            padding: 1rem 1.5rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            border-left: 6px solid #ec4899;
            box-shadow: 0 4px 15px rgba(236, 72, 153, 0.1);
            font-weight: 600;
        }
        
        /* Input borders */
        div[data-baseweb="input"] {
            border-radius: 20px !important;
            border: 2px solid #fbcfe8 !important;
            background-color: white !important;
        }
        div[data-baseweb="input"]:focus-within {
            border-color: #ec4899 !important;
            box-shadow: 0 0 0 1px #ec4899 !important;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: rgba(255, 255, 255, 0.6);
            backdrop-filter: blur(10px);
            border-right: 1px solid #fce7f3;
        }
    </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 6])
    with col1:
        # A chic, modern, elegant AI/brain spark icon
        st.image("https://cdn-icons-png.flaticon.com/512/8652/8652695.png", width=95)
    with col2:
        st.title("DocuQuery ✨")
        st.markdown("<p class='subtitle'>Your Smart & Elegant <strong>AI Document Assistant</strong></p>", unsafe_allow_html=True)

    api_key = os.getenv("GOOGLE_API_KEY")

    with st.sidebar:
        st.markdown("## 🎀 Drop Your Files")
        st.markdown("Upload PDFs to weave them into the AI.")
        uploaded_pdfs = st.file_uploader("", accept_multiple_files=True, type=["pdf"])
        
        if st.button("✨ Process Documents ✨"):
            if not api_key:
                st.error("🔑 Missing Google API Key in settings.")
            elif not uploaded_pdfs:
                st.warning("📄 Please upload a document to begin.")
            else:
                with st.spinner("Extracting text & building FAISS Vector Database..."):
                    text_chunks, metadatas = extract_text_and_metadatas_from_pdfs(uploaded_pdfs)
                    if not text_chunks:
                        st.error("No readable text found in the PDFs.")
                    else:
                        create_vector_db(text_chunks, metadatas)
                        st.success("✅ Multi-document database built! Ready for cross-referencing queries.")

    st.markdown("---")
    
    if os.path.exists("faiss_index_store"):
        st.markdown("### 💭 Let's chat about your document!")
        query = st.chat_input("Ask a complex question about your documents...")
        if query:
            st.markdown(f"**You asked:** *{query}*")
            process_query(query)
    else:
        st.info("👈 Please start the magic by uploading a document in the sidebar.")

if __name__ == "__main__":
    main()
