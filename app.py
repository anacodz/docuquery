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

    Previous Conversation History:
    {chat_history}

    Context:
    {context}
    
    Question: 
    {question}

    Answer:
    """
    
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.2)
    prompt = PromptTemplate(template=template, input_variables=["context", "question", "chat_history"])
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)

def process_query(query):
    # Build history string BEFORE appending the new user query to context
    chat_history_str = ""
    for msg in st.session_state.messages[-4:]: # Only keep last 4 context items
        role = "System" if msg["role"] == "assistant" else "User"
        chat_history_str += f"{role}: {msg['content']}\n"

    # Add user query to state so it renders correctly going forward
    st.session_state.messages.append({"role": "user", "content": query})

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    try:
        db = FAISS.load_local("faiss_index_store", embeddings, allow_dangerous_deserialization=True)
        # Fetch more context (12 chunks) so the AI has enough text to compare multiple documents
        relevant_docs = db.similarity_search(query, k=12)

        chain = setup_qa_chain()
        with st.chat_message("assistant"):
            with st.spinner("🧠 Synthesizing answer from multiple sources..."):
                res = chain.invoke(
                    {"input_documents": relevant_docs, "question": query, "chat_history": chat_history_str}
                )
            
            answer = res["output_text"]
            st.markdown(answer)
            
            with st.expander("📚 View Extracted Source Context"):
                for i, doc in enumerate(relevant_docs[:8]): # Only show top 8 in UI to avoid clutter
                    source_file = doc.metadata.get('source', 'Unknown Document')
                    st.markdown(f"**Source: {source_file}**")
                    st.markdown(f"> *{doc.page_content}*")
                    st.divider()
                    
            # Save AI response to session state memory
            st.session_state.messages.append({"role": "assistant", "content": answer, "docs": relevant_docs})
            
    except Exception as e:
        st.error(f"Something went wrong. Is the API key configured properly? Error: {str(e)}")

def main():
    st.set_page_config(page_title="DocuQuery | AI PDF Assistant", page_icon="🎀", layout="wide")
    
    # 🎨 Color Theme Selector in Sidebar
    # left sidebar theme picker
    with st.sidebar:
        st.markdown("### 🎨 Choose a Theme")
        color_theme = st.selectbox("", ["Elegant Pink", "Ocean Blue", "Midnight Dark", "Forest Green"])

    # Define color mappings based on selection
    if color_theme == "Elegant Pink":
        bg_gradient = "linear-gradient(135deg, #fff0f5 0%, #f3e8ff 100%)"
        text_color = "#4a044e"
        btn_gradient = "linear-gradient(45deg, #ec4899 0%, #8b5cf6 100%)"
        accent = "#ec4899"
        sidebar_bg = "rgba(255, 255, 255, 0.6)"
        input_bg = "white"
        panel_bg = "rgba(255, 255, 255, 0.85)"
    elif color_theme == "Ocean Blue":
        bg_gradient = "linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)"
        text_color = "#0c4a6e"
        btn_gradient = "linear-gradient(45deg, #0ea5e9 0%, #3b82f6 100%)"
        accent = "#0ea5e9"
        sidebar_bg = "rgba(255, 255, 255, 0.6)"
        input_bg = "white"
        panel_bg = "rgba(255, 255, 255, 0.85)"
    elif color_theme == "Midnight Dark":
        bg_gradient = "linear-gradient(135deg, #0f172a 0%, #1e293b 100%)"
        text_color = "#f8fafc"
        btn_gradient = "linear-gradient(45deg, #6366f1 0%, #8b5cf6 100%)"
        accent = "#8b5cf6"
        sidebar_bg = "rgba(15, 23, 42, 0.8)"
        input_bg = "#1e293b"
        panel_bg = "rgba(30, 41, 59, 0.85)"
    else: # Forest Green
        bg_gradient = "linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)"
        text_color = "#14532d"
        btn_gradient = "linear-gradient(45deg, #22c55e 0%, #10b981 100%)"
        accent = "#22c55e"
        sidebar_bg = "rgba(255, 255, 255, 0.6)"
        input_bg = "white"
        panel_bg = "rgba(255, 255, 255, 0.85)"

    # Dynamic Custom CSS 
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');
        
        /* Force background gradient on Streamlit main container */
        .stApp {{
            background: {bg_gradient} !important;
        }}
        
        [data-testid="stHeader"] {{
            background: transparent !important;
        }}
        
        /* Fix Chat Input overlapping text when scrolling */
        [data-testid="stBottomBlockContainer"], [data-testid="stBottom"] {{
            background: {bg_gradient} !important;
            border-top: 1px solid rgba(128, 128, 128, 0.2);
            z-index: 9999 !important;
        }}
        
        /* Global font and color replacements */
        p, h1, h2, h3, h4, h5, h6, span, label, li {{
            font-family: 'Poppins', sans-serif !important;
        }}
        
        h1, h2, h3, p, label, .markdown-text-container, .stMarkdown {{
            color: {text_color} !important;
        }}
        
        /* Button Styling */
        .stButton>button {{
            background: {btn_gradient} !important;
            color: white !important;
            border-radius: 30px !important;
            border: none !important;
            padding: 0.6rem 1.5rem !important;
            font-weight: 600 !important;
            transition: all 0.4s ease !important;
            width: 100% !important;
        }}
        .stButton>button:hover {{
            transform: translateY(-3px) scale(1.02) !important;
            filter: brightness(1.1) !important;
        }}
        
        /* Success Message Container */
        .success-message {{
            background: {panel_bg} !important;
            backdrop-filter: blur(10px) !important;
            color: {text_color} !important;
            padding: 1rem 1.5rem !important;
            border-radius: 12px !important;
            margin-bottom: 1rem !important;
            border-left: 6px solid {accent} !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
            font-weight: 600 !important;
        }}
        
        /* Expander headers */
        .streamlit-expanderHeader {{
            background-color: transparent !important;
            color: {text_color} !important;
        }}
        [data-testid="stExpander"] {{
            background-color: {panel_bg} !important;
            border-radius: 10px !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
        }}
        
        /* Input widget borders and backgrounds */
        div[data-baseweb="input"], div[data-baseweb="select"] {{
            border-radius: 20px !important;
            border: 2px solid {accent} !important;
            background-color: {input_bg} !important;
        }}
        div[data-baseweb="input"] input, div[data-baseweb="select"] div {{
            color: {text_color} !important;
            background-color: transparent !important;
        }}
        
        /* File Uploader area */
        [data-testid="stFileUploadDropzone"] {{
            background-color: {panel_bg} !important;
            border: 2px dashed {accent} !important;
        }}
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {{
            background-color: {sidebar_bg} !important;
            backdrop-filter: blur(10px) !important;
            border-right: 1px solid rgba(255,255,255,0.1) !important;
        }}
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
                        st.session_state.messages = [] # Clear memory for a completely new DB context!
                        st.success("✅ Multi-document database built! Ready for cross-referencing queries.")

    st.markdown("---")
    
    # Initialize chat history memory
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if os.path.exists("faiss_index_store"):
        st.markdown("### 💭 Let's chat about your document!")
        
        # Render historical chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "docs" in message:
                    with st.expander("📚 View Extracted Source Context"):
                        for i, doc in enumerate(message["docs"][:8]):
                            source_file = doc.metadata.get('source', 'Unknown Document')
                            st.markdown(f"**Source: {source_file}**")
                            st.markdown(f"> *{doc.page_content}*")
                            st.divider()

        # Capture a new query
        query = st.chat_input("Ask a complex question about your documents...")
        if query:
            with st.chat_message("user"):
                st.markdown(query)
            process_query(query)
    else:
        st.info("👈 Please start the magic by uploading a document in the sidebar.")

if __name__ == "__main__":
    main()
