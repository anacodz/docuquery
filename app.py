import streamlit as st
import os
import sqlite3
import json
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

# --- DATABASE SETUP ---
DB_NAME = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  role TEXT, 
                  content TEXT, 
                  docs_json TEXT)''')
    conn.commit()
    conn.close()

def load_chat_history():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT role, content, docs_json FROM messages ORDER BY id ASC')
    rows = c.fetchall()
    conn.close()
    
    messages = []
    for role, content, docs_json in rows:
        msg = {"role": role, "content": content}
        if docs_json:
            msg["docs"] = json.loads(docs_json)
        messages.append(msg)
    return messages

def save_message(role, content, docs=None):
    docs_json = None
    if docs:
        docs_json = json.dumps([{"source": d.metadata.get("source", "Unknown"), "content": d.page_content} for d in docs])
        
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('INSERT INTO messages (role, content, docs_json) VALUES (?, ?, ?)', (role, content, docs_json))
    conn.commit()
    conn.close()

def clear_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('DELETE FROM messages')
    conn.commit()
    conn.close()
    st.session_state.messages = []

# Initialize database on app start
init_db()

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
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
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
    for msg in st.session_state.messages[-4:]: 
        role = "System" if msg["role"] == "assistant" else "User"
        chat_history_str += f"{role}: {msg['content']}\n"

    # Add user query to state and Database
    st.session_state.messages.append({"role": "user", "content": query})
    save_message("user", query)

    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    
    try:
        db = FAISS.load_local("faiss_index_store", embeddings, allow_dangerous_deserialization=True)
        # Fetch more context (12 chunks) so the AI has enough text to compare multiple documents
        relevant_docs = db.similarity_search(query, k=12)

        chain = setup_qa_chain()
        with st.chat_message("assistant", avatar="✨"):
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
                    
            # Save AI response to session state and SQLite Database
            st.session_state.messages.append({"role": "assistant", "content": answer, "docs": [d.__dict__ for d in relevant_docs]})
            save_message("assistant", answer, relevant_docs)
            
    except Exception as e:
        st.error(f"Something went wrong. Is the API key configured properly? Error: {str(e)}")


def get_theme(color_theme):
    """Returns a dict of theme variables for the selected color theme."""
    themes = {
        "Elegant Pink": {
            "bg_gradient": "linear-gradient(135deg, #fff0f5 0%, #fce7f3 30%, #f3e8ff 100%)",
            "text_color": "#4a044e",
            "text_secondary": "#7c3aed",
            "btn_gradient": "linear-gradient(135deg, #ec4899 0%, #a855f7 50%, #8b5cf6 100%)",
            "accent": "#ec4899",
            "accent_light": "rgba(236, 72, 153, 0.12)",
            "sidebar_bg": "rgba(255, 255, 255, 0.65)",
            "input_bg": "rgba(255, 255, 255, 0.9)",
            "panel_bg": "rgba(255, 255, 255, 0.75)",
            "chat_user_bg": "rgba(236, 72, 153, 0.08)",
            "chat_ai_bg": "rgba(139, 92, 246, 0.08)",
            "card_border": "rgba(236, 72, 153, 0.2)",
            "divider": "rgba(236, 72, 153, 0.15)",
            "shadow": "rgba(236, 72, 153, 0.1)",
        },
        "Ocean Blue": {
            "bg_gradient": "linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 30%, #dbeafe 100%)",
            "text_color": "#0c4a6e",
            "text_secondary": "#0369a1",
            "btn_gradient": "linear-gradient(135deg, #0ea5e9 0%, #3b82f6 50%, #6366f1 100%)",
            "accent": "#0ea5e9",
            "accent_light": "rgba(14, 165, 233, 0.12)",
            "sidebar_bg": "rgba(255, 255, 255, 0.65)",
            "input_bg": "rgba(255, 255, 255, 0.9)",
            "panel_bg": "rgba(255, 255, 255, 0.75)",
            "chat_user_bg": "rgba(14, 165, 233, 0.08)",
            "chat_ai_bg": "rgba(59, 130, 246, 0.08)",
            "card_border": "rgba(14, 165, 233, 0.2)",
            "divider": "rgba(14, 165, 233, 0.15)",
            "shadow": "rgba(14, 165, 233, 0.1)",
        },
        "Midnight Dark": {
            "bg_gradient": "linear-gradient(135deg, #0f172a 0%, #1e1b4b 30%, #1e293b 100%)",
            "text_color": "#f8fafc",
            "text_secondary": "#a5b4fc",
            "btn_gradient": "linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%)",
            "accent": "#8b5cf6",
            "accent_light": "rgba(139, 92, 246, 0.15)",
            "sidebar_bg": "rgba(15, 23, 42, 0.85)",
            "input_bg": "rgba(30, 41, 59, 0.9)",
            "panel_bg": "rgba(30, 41, 59, 0.75)",
            "chat_user_bg": "rgba(99, 102, 241, 0.12)",
            "chat_ai_bg": "rgba(139, 92, 246, 0.12)",
            "card_border": "rgba(139, 92, 246, 0.25)",
            "divider": "rgba(139, 92, 246, 0.2)",
            "shadow": "rgba(0, 0, 0, 0.3)",
        },
        "Forest Green": {
            "bg_gradient": "linear-gradient(135deg, #f0fdf4 0%, #dcfce7 30%, #d1fae5 100%)",
            "text_color": "#14532d",
            "text_secondary": "#15803d",
            "btn_gradient": "linear-gradient(135deg, #22c55e 0%, #10b981 50%, #059669 100%)",
            "accent": "#22c55e",
            "accent_light": "rgba(34, 197, 94, 0.12)",
            "sidebar_bg": "rgba(255, 255, 255, 0.65)",
            "input_bg": "rgba(255, 255, 255, 0.9)",
            "panel_bg": "rgba(255, 255, 255, 0.75)",
            "chat_user_bg": "rgba(34, 197, 94, 0.08)",
            "chat_ai_bg": "rgba(16, 185, 129, 0.08)",
            "card_border": "rgba(34, 197, 94, 0.2)",
            "divider": "rgba(34, 197, 94, 0.15)",
            "shadow": "rgba(34, 197, 94, 0.1)",
        },
    }
    return themes.get(color_theme, themes["Elegant Pink"])


def inject_css(t):
    """Inject the full dynamic CSS using the theme dict `t`."""
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        /* ===== GLOBAL RESET & BASE ===== */
        *, *::before, *::after {{
            box-sizing: border-box;
        }}
        
        .stApp {{
            background: {t["bg_gradient"]} !important;
            background-size: 400% 400% !important;
            animation: gradient-shift 20s ease infinite !important;
        }}
        
        @keyframes gradient-shift {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
        
        [data-testid="stHeader"] {{
            background: transparent !important;
        }}
        
        /* ===== TYPOGRAPHY ===== */
        html, body, p, h1, h2, h3, h4, h5, h6, label, li, span, div, input, textarea, button {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        }}
        
        h1, h2, h3, p, label, .markdown-text-container, .stMarkdown {{
            color: {t["text_color"]} !important;
        }}
        
        /* ===== CHAT INPUT BOTTOM BAR ===== */
        [data-testid="stBottomBlockContainer"], [data-testid="stBottom"] {{
            background: {t["bg_gradient"]} !important;
            border-top: 1px solid {t["divider"]};
            z-index: 9999 !important;
            padding-top: 0.75rem !important;
        }}
        
        [data-testid="stChatInput"] {{
            border-radius: 16px !important;
            overflow: hidden !important;
        }}
        
        [data-testid="stChatInput"] textarea {{
            font-size: 0.95rem !important;
            padding: 0.85rem 1rem !important;
        }}

        /* ===== CHAT MESSAGE BUBBLES ===== */
        [data-testid="stChatMessage"] {{
            background: {t["panel_bg"]} !important;
            backdrop-filter: blur(16px) saturate(180%) !important;
            -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
            border: 1px solid {t["card_border"]} !important;
            border-radius: 16px !important;
            padding: 1.1rem 1.3rem !important;
            margin-bottom: 0.85rem !important;
            box-shadow: 0 4px 24px {t["shadow"]}, 0 1px 3px rgba(0,0,0,0.04) !important;
            animation: msgFadeIn 0.4s ease-out !important;
            transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        }}
        
        [data-testid="stChatMessage"]:hover {{
            transform: translateY(-1px) !important;
            box-shadow: 0 8px 32px {t["shadow"]}, 0 2px 6px rgba(0,0,0,0.06) !important;
        }}
        
        @keyframes msgFadeIn {{
            from {{ opacity: 0; transform: translateY(12px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        /* Override default chat message avatar SVGs */
        [data-testid="stChatMessage"] .st-emotion-cache-1c7y2kd {{
            display: none !important;
        }}

        /* ===== FLOATING LOGO & PULSING TITLE ===== */
        .floating-logo {{
            animation: float 5s ease-in-out infinite;
            filter: drop-shadow(0 4px 12px {t["shadow"]});
        }}
        @keyframes float {{
            0% {{ transform: translateY(0px) rotate(0deg); }}
            25% {{ transform: translateY(-6px) rotate(1deg); }}
            50% {{ transform: translateY(-10px) rotate(0deg); }}
            75% {{ transform: translateY(-6px) rotate(-1deg); }}
            100% {{ transform: translateY(0px) rotate(0deg); }}
        }}
        
        /* ===== CENTERED HEADER ===== */
        .app-header {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 1.2rem;
            padding: 1.5rem 0 0.5rem 0;
        }}
        
        .app-title {{
            font-size: 2.6rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.03em !important;
            background: {t["btn_gradient"]};
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
            line-height: 1.2 !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        
        .app-subtitle {{
            font-size: 1.05rem !important;
            font-weight: 400 !important;
            color: {t["text_secondary"]} !important;
            text-align: center !important;
            margin: 0.25rem 0 0 0 !important;
            padding: 0 !important;
            opacity: 0.85;
        }}
        
        /* ===== BUTTONS ===== */
        .stButton>button {{
            background: {t["btn_gradient"]} !important;
            background-size: 200% 200% !important;
            color: white !important;
            border-radius: 14px !important;
            border: none !important;
            padding: 0.65rem 1.5rem !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            letter-spacing: 0.01em !important;
            transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1) !important;
            width: 100% !important;
            box-shadow: 0 4px 14px {t["shadow"]} !important;
        }}
        .stButton>button:hover {{
            transform: translateY(-2px) scale(1.01) !important;
            box-shadow: 0 8px 25px {t["shadow"]} !important;
            background-position: right center !important;
            filter: brightness(1.08) !important;
        }}
        .stButton>button:active {{
            transform: translateY(0px) scale(0.99) !important;
        }}
        
        /* ===== EXPANDER (Source Context) ===== */
        .streamlit-expanderHeader {{
            background-color: transparent !important;
            color: {t["text_color"]} !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
        }}
        [data-testid="stExpander"] {{
            background-color: {t["panel_bg"]} !important;
            backdrop-filter: blur(12px) !important;
            border-radius: 12px !important;
            border: 1px solid {t["card_border"]} !important;
            margin-top: 0.5rem !important;
        }}
        
        /* ===== INPUTS & SELECTS ===== */
        div[data-baseweb="input"], div[data-baseweb="select"] {{
            border-radius: 12px !important;
            border: 1.5px solid {t["card_border"]} !important;
            background-color: {t["input_bg"]} !important;
            transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
        }}
        div[data-baseweb="input"]:focus-within, div[data-baseweb="select"]:focus-within {{
            border-color: {t["accent"]} !important;
            box-shadow: 0 0 0 3px {t["accent_light"]} !important;
        }}
        div[data-baseweb="input"] input, div[data-baseweb="select"] div {{
            color: {t["text_color"]} !important;
            background-color: transparent !important;
        }}
        
        /* ===== FILE UPLOADER (Cleaned) ===== */
        [data-testid="stFileUploadDropzone"] {{
            background-color: {t["panel_bg"]} !important;
            backdrop-filter: blur(8px) !important;
            border: 2px dashed {t["card_border"]} !important;
            border-radius: 14px !important;
            padding: 1.5rem 1rem !important;
            transition: border-color 0.3s ease, background-color 0.3s ease !important;
        }}
        [data-testid="stFileUploadDropzone"]:hover {{
            border-color: {t["accent"]} !important;
            background-color: {t["accent_light"]} !important;
        }}
        
        /* Hide the massive cloud icon to save space */
        [data-testid="stFileUploadDropzone"] svg {{
            display: none !important;
        }}
        
        /* Ensure the wrapper uses flex properly to prevent overlap */
        [data-testid="stFileUploadDropzone"] > div {{
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            gap: 12px !important;
        }}

        /* Style the internal button */
        [data-testid="stFileUploadDropzone"] button {{
            background: {t["accent_light"]} !important;
            color: {t["accent"]} !important;
            border: 1.5px solid {t["accent"]} !important;
            border-radius: 10px !important;
            padding: 0.5rem 1rem !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            width: 100% !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
        }}
        [data-testid="stFileUploadDropzone"] button:hover {{
            background: {t["accent"]} !important;
            color: white !important;
        }}
        
        /* File size limit text */
        [data-testid="stFileUploadDropzone"] small {{
            font-size: 0.75rem !important;
            opacity: 0.6 !important;
            color: {t["text_color"]} !important;
            text-align: center !important;
        }}
        
        /* ===== SIDEBAR ===== */
        [data-testid="stSidebar"] {{
            background-color: {t["sidebar_bg"]} !important;
            backdrop-filter: blur(20px) saturate(180%) !important;
            -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
            border-right: 1px solid {t["card_border"]} !important;
        }}
        
        [data-testid="stSidebar"] .stMarkdown h2 {{
            font-size: 1.15rem !important;
            font-weight: 700 !important;
        }}
        
        [data-testid="stSidebar"] .stMarkdown h3 {{
            font-size: 0.95rem !important;
            font-weight: 600 !important;
        }}
        
        /* ===== DIVIDER ===== */
        hr {{
            border: none !important;
            height: 1px !important;
            background: {t["divider"]} !important;
            margin: 1rem 0 !important;
        }}
        
        /* ===== CUSTOM COMPONENTS ===== */
        
        /* Welcome hero section */
        .welcome-hero {{
            text-align: center;
            padding: 3rem 1.5rem;
            animation: heroFadeIn 0.8s ease-out;
        }}
        @keyframes heroFadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .welcome-icon {{
            font-size: 4rem;
            margin-bottom: 1rem;
            display: inline-block;
            animation: float 5s ease-in-out infinite;
        }}
        
        .welcome-title {{
            font-size: 1.8rem !important;
            font-weight: 700 !important;
            color: {t["text_color"]} !important;
            margin-bottom: 0.5rem !important;
        }}
        
        .welcome-desc {{
            font-size: 1.05rem !important;
            color: {t["text_secondary"]} !important;
            max-width: 520px;
            margin: 0 auto 2rem auto;
            line-height: 1.6;
        }}
        
        /* Feature cards */
        .feature-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            max-width: 720px;
            margin: 0 auto;
        }}
        
        .feature-card {{
            background: {t["panel_bg"]};
            backdrop-filter: blur(12px);
            border: 1px solid {t["card_border"]};
            border-radius: 14px;
            padding: 1.25rem 1rem;
            text-align: center;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        .feature-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 12px 32px {t["shadow"]};
        }}
        .feature-card .feat-icon {{
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
        }}
        .feature-card .feat-title {{
            font-size: 0.85rem;
            font-weight: 700;
            color: {t["text_color"]};
            margin-bottom: 0.25rem;
        }}
        .feature-card .feat-desc {{
            font-size: 0.75rem;
            color: {t["text_secondary"]};
            line-height: 1.4;
        }}
        
        /* Sidebar stat badges */
        .sidebar-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            background: {t["accent_light"]};
            border: 1px solid {t["card_border"]};
            border-radius: 20px;
            padding: 0.3rem 0.75rem;
            font-size: 0.78rem;
            font-weight: 600;
            color: {t["text_color"]};
            margin-right: 0.4rem;
            margin-bottom: 0.4rem;
        }}
        
        /* Sidebar section label */
        .sidebar-section {{
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {t["text_secondary"]};
            margin-bottom: 0.5rem;
            margin-top: 1rem;
            opacity: 0.7;
        }}
        
        /* Chat header bar */
        .chat-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.5rem 0;
            margin-bottom: 0.25rem;
        }}
        
        .chat-header-title {{
            font-size: 1.1rem !important;
            font-weight: 700 !important;
            color: {t["text_color"]} !important;
            margin: 0 !important;
        }}
        
        /* Powered by footer */
        .powered-by {{
            text-align: center;
            font-size: 0.65rem;
            color: {t["text_secondary"]};
            opacity: 0.5;
            padding: 0.5rem 0 0 0;
            letter-spacing: 0.05em;
        }}
        
        /* Source context styling inside expanders */
        .source-chip {{
            display: inline-block;
            background: {t["accent_light"]};
            border: 1px solid {t["card_border"]};
            border-radius: 8px;
            padding: 0.2rem 0.6rem;
            font-size: 0.75rem;
            font-weight: 600;
            color: {t["accent"]};
            margin-bottom: 0.4rem;
        }}
        
        /* Streamlit toast/alert overrides */
        [data-testid="stAlert"] {{
            border-radius: 12px !important;
            backdrop-filter: blur(8px) !important;
        }}
    </style>
    """, unsafe_allow_html=True)


def render_welcome(t):
    """Render an animated welcome/empty state with feature highlights."""
    st.markdown(f"""
    <div style="display:flex; justify-content:center; align-items:center; width:100%;">
        <div class="welcome-hero">
            <div class="welcome-icon">📄</div>
            <div class="welcome-title">Welcome to DocuQuery</div>
            <p class="welcome-desc">
                Upload your PDF documents and start asking questions. 
                The AI reads, understands, and cross-references your files 
                to give you precise, source-cited answers.
            </p>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feat-icon">🌍</div>
                    <div class="feat-title">50+ Languages</div>
                    <div class="feat-desc">Multilingual embeddings understand documents in any language</div>
                </div>
                <div class="feature-card">
                    <div class="feat-icon">📑</div>
                    <div class="feat-title">Multi-Document</div>
                    <div class="feat-desc">Upload multiple PDFs and cross-reference across all of them</div>
                </div>
                <div class="feature-card">
                    <div class="feat-icon">🔒</div>
                    <div class="feat-title">Local Processing</div>
                    <div class="feat-desc">Embeddings generated locally — your data stays private</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar(t):
    """Build the full sidebar UI. Returns (uploaded_pdfs, process_clicked)."""
    with st.sidebar:
        # Theme selector
        st.markdown('<div class="sidebar-section">Appearance</div>', unsafe_allow_html=True)
        color_theme = st.selectbox("Theme", ["Elegant Pink", "Ocean Blue", "Midnight Dark", "Forest Green"], label_visibility="collapsed")
        
        st.markdown("---")
        
        # Document upload section
        st.markdown("## 📄 Documents")
        st.markdown('<p style="font-size:0.85rem; opacity:0.7;">Upload PDFs to build your knowledge base.</p>', unsafe_allow_html=True)
        
        uploaded_pdfs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"], label_visibility="collapsed")
        
        # Show uploaded file count badge
        if uploaded_pdfs:
            badge_html = f'<span class="sidebar-badge">📎 {len(uploaded_pdfs)} file{"s" if len(uploaded_pdfs) != 1 else ""} selected</span>'
            st.markdown(badge_html, unsafe_allow_html=True)
        
        process_clicked = st.button("⚡ Process Documents")
        
        # Show processing stats if available
        if "doc_stats" in st.session_state and st.session_state.doc_stats:
            stats = st.session_state.doc_stats
            st.markdown("---")
            st.markdown('<div class="sidebar-section">Knowledge Base</div>', unsafe_allow_html=True)
            stats_html = (
                f'<span class="sidebar-badge">📄 {stats["docs"]} doc{"s" if stats["docs"] != 1 else ""}</span>'
                f'<span class="sidebar-badge">🧩 {stats["chunks"]} chunks</span>'
            )
            st.markdown(stats_html, unsafe_allow_html=True)
        
        # Footer
        st.markdown("---")
        st.markdown(
            '<div class="powered-by">BUILT WITH LANGCHAIN · FAISS · GEMINI</div>',
            unsafe_allow_html=True
        )
        
    return color_theme, uploaded_pdfs, process_clicked


def main():
    st.set_page_config(page_title="DocuQuery | AI PDF Assistant", page_icon="📄", layout="wide")
    
    # ── Sidebar (single block to avoid duplicate widgets) ──
    with st.sidebar:
        st.markdown('<div class="sidebar-section">Appearance</div>', unsafe_allow_html=True)
        color_theme = st.selectbox("Theme", ["Elegant Pink", "Ocean Blue", "Midnight Dark", "Forest Green"], label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown("## 📄 Documents")
        st.markdown('<p style="font-size:0.85rem; opacity:0.7;">Upload PDFs to build your knowledge base.</p>', unsafe_allow_html=True)
        
        uploaded_pdfs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"], label_visibility="collapsed")
        
        if uploaded_pdfs:
            badge_html = f'<span class="sidebar-badge">📎 {len(uploaded_pdfs)} file{"s" if len(uploaded_pdfs) != 1 else ""} selected</span>'
            st.markdown(badge_html, unsafe_allow_html=True)
        
        process_clicked = st.button("⚡ Process Documents")
        
        # Show stats if available
        if "doc_stats" in st.session_state and st.session_state.doc_stats:
            stats = st.session_state.doc_stats
            st.markdown("---")
            st.markdown('<div class="sidebar-section">Knowledge Base</div>', unsafe_allow_html=True)
            stats_html = (
                f'<span class="sidebar-badge">📄 {stats["docs"]} doc{"s" if stats["docs"] != 1 else ""}</span>'
                f'<span class="sidebar-badge">🧩 {stats["chunks"]} chunks</span>'
            )
            st.markdown(stats_html, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown(
            '<div class="powered-by">BUILT WITH LANGCHAIN · FAISS · GEMINI</div>',
            unsafe_allow_html=True
        )
    
    # Get theme and inject CSS immediately
    t = get_theme(color_theme)
    inject_css(t)
    
    # ── Header (centered with HTML flexbox) ──
    st.markdown(f"""
    <div class="app-header">
        <img src="https://cdn-icons-png.flaticon.com/512/8652/8652695.png" class="floating-logo" width="70">
        <div>
            <h1 class="app-title">DocuQuery ✨</h1>
            <p class="app-subtitle">Your Smart & Elegant <strong>AI Document Assistant</strong></p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # ── Handle document processing ──
    if process_clicked:
        if not api_key:
            st.error("🔑 Missing Google API Key in settings.")
        elif not uploaded_pdfs:
            st.warning("📄 Please upload at least one document.")
        else:
            with st.spinner("Extracting text & building vector database..."):
                text_chunks, metadatas = extract_text_and_metadatas_from_pdfs(uploaded_pdfs)
                if not text_chunks:
                    st.error("No readable text found in the PDFs.")
                else:
                    create_vector_db(text_chunks, metadatas)
                    clear_db()  # Clear DB for a completely new context
                    st.session_state.doc_stats = {
                        "docs": len(uploaded_pdfs),
                        "chunks": len(text_chunks),
                    }
                    st.success(f"✅ Knowledge base built — {len(text_chunks)} chunks from {len(uploaded_pdfs)} document{'s' if len(uploaded_pdfs) != 1 else ''}. Ready to query!")

    st.markdown("---")
    
    # ── Initialize memory from SQLite ──
    if "messages" not in st.session_state:
        st.session_state.messages = load_chat_history()
    
    # ── Main content area ──
    if os.path.exists("faiss_index_store"):
        # Chat header with clear button
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown("### 💭 Chat with your documents")
        with col2:
            if st.button("🗑️ Clear", help="Clear conversation history"):
                clear_db()
                st.rerun()
                
        # Render historical chat messages from DB
        for message in st.session_state.messages:
            avatar_icon = "👤" if message["role"] == "user" else "✨"
            with st.chat_message(message["role"], avatar=avatar_icon):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "docs" in message:
                    with st.expander("📚 View Source Context"):
                        for i, doc in enumerate(message["docs"][:8]):
                            source_file = doc.get("source", "Unknown Document") if isinstance(doc, dict) else doc.metadata.get('source', 'Unknown Document')
                            doc_content = doc.get("content", "") if isinstance(doc, dict) else doc.page_content
                            st.markdown(f'<span class="source-chip">📎 {source_file}</span>', unsafe_allow_html=True)
                            st.markdown(f"> *{doc_content}*")
                            st.divider()

        # Capture a new query
        query = st.chat_input("Ask anything about your documents...")
        if query:
            with st.chat_message("user", avatar="👤"):
                st.markdown(query)
            process_query(query)
    else:
        render_welcome(t)

if __name__ == "__main__":
    main()
