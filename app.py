import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
        except Exception as e:
            st.error(f"Error reading file {pdf.name}: {e}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    # Uses HuggingFace all-MiniLM-L6-v2 which runs locally and is open-source
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are an intelligent enterprise AI assistant designed to extract insights from technical documents.
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, politely say, "The requested information is not available in the given document.", and do not hallucinate an answer.

    Context:
    {context}
    
    Question: 
    {question}

    Answer:
    """
    
    # We use Google's free Gemini tier (must supply API key via .env)
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    # In a production app, embeddings model should be loaded once globally
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question, k=4)

        chain = get_conversational_chain()
        response = chain.invoke(
            {"input_documents": docs, "question": user_question}
        )
        st.markdown("### Answer:")
        st.write(response["output_text"])
        
        with st.expander("View Source Document Chunks"):
            for i, doc in enumerate(docs):
                st.write(f"**Chunk {i+1}**:")
                st.write(doc.page_content)
    except Exception as e:
        st.error("An error occurred during response generation. Make sure your API key is correct.")
        st.exception(e)

def main():
    st.set_page_config(page_title="Enterprise RAG Assistant", page_icon="📄", layout="wide")
    st.title("Enterprise RAG Document AI Assistant 📄🤖")
    st.markdown("This application uses **Retrieval-Augmented Generation (RAG)** to answer questions securely based on the PDFs you upload.")

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    with st.sidebar:
        st.title("📁 Document Upload")
        st.markdown("Upload PDFs to build the vector database.")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        
        if st.button("Submit & Process Docs"):
            if not api_key or api_key == "your_api_key_here":
                st.error("Please add your Google API Key to the .env file.")
            elif not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Parsing text & Generating Embeddings..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    if not text_chunks:
                        st.error("No extractable text found in these documents.")
                    else:
                        get_vector_store(text_chunks)
                        st.success("Vector Database Built Successfully!")

    # Check if a vector DB exists locally to enable searching
    if os.path.exists("faiss_index"):
        st.markdown("---")
        user_question = st.text_input("Ask a question about the uploaded document(s):")
        if user_question:
            user_input(user_question)
    else:
        st.info("👈 Please upload a document and process it from the sidebar to start asking questions.")

if __name__ == "__main__":
    main()
