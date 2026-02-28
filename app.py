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
        # A chic, visually dashing icon
        st.image("https://cdn-icons-png.flaticon.com/512/9307/9307767.png", width=95)
    with col2:
        st.title("DocuQuery ✨")
        st.markdown("<p class='subtitle'>Your Smart & Elegant <strong>AI Document Assistant</strong> 💅</p>", unsafe_allow_html=True)

    api_key = os.getenv("GOOGLE_API_KEY")

    with st.sidebar:
        st.markdown("## 🎀 Drop Your Files")
        st.markdown("Upload PDFs to weave them into the AI.")
        uploaded_pdfs = st.file_uploader("", accept_multiple_files=True)
        
        if st.button("✨ Process Documents ✨"):
            if not api_key:
                st.error("🔑 Missing Google API Key in settings.")
            elif not uploaded_pdfs:
                st.warning("📄 Please upload a file first darling.")
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
        st.markdown("### 💭 Let's chat about your document!")
        query = st.chat_input("Ask a complex question about your documents...")
        if query:
            st.markdown(f"**You asked:** *{query}*")
            process_query(query)
    else:
        st.info("👈 Please start the magic by uploading a document in the sidebar.")

if __name__ == "__main__":
    main()
