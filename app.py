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
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)

def process_query(query):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    try:
        db = FAISS.load_local("faiss_index_store", embeddings, allow_dangerous_deserialization=True)
        relevant_docs = db.similarity_search(query, k=4)

        chain = setup_qa_chain()
        res = chain.invoke(
            {"input_documents": relevant_docs, "question": query}
        )
        st.write("### Answer:")
        st.write(res["output_text"])
        
        with st.expander("Show references"):
            for i, doc in enumerate(relevant_docs):
                st.write(f"**Source {i+1}**:")
                st.write(doc.page_content)
    except Exception as e:
        st.error(f"Something went wrong. Is the API key configured properly? Error: {str(e)}")

def main():
    st.set_page_config(page_title="DocuQuery | PDF Assistant", layout="wide")
    st.title("DocuQuery")
    st.markdown("A simple RAG tool I built to query information from large PDFs using LangChain and Gemini.")

    api_key = os.getenv("GOOGLE_API_KEY")

    with st.sidebar:
        st.header("Upload Documents")
        uploaded_pdfs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)
        
        if st.button("Process Documents"):
            if not api_key:
                st.error("Missing Google API Key in .env file.")
            elif not uploaded_pdfs:
                st.warning("Please upload a file first.")
            else:
                with st.spinner("Extracting and processing text..."):
                    raw_text = extract_text_from_pdfs(uploaded_pdfs)
                    chunks = split_text_into_chunks(raw_text)
                    if not chunks:
                        st.error("No readable text found in the PDF.")
                    else:
                        create_vector_db(chunks)
                        st.success("Documents processed and database created!")

    if os.path.exists("faiss_index_store"):
        st.markdown("---")
        query = st.text_input("What do you want to know about the documents?")
        if query:
            process_query(query)
    else:
        st.info("Upload and process some documents in the sidebar to get started.")

if __name__ == "__main__":
    main()
