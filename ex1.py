import os
import streamlit as st
# *** Change 1: Import the better loader ***
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# --- Configuration ---
VECTOR_DB_PATH = "chroma_db"
OLLAMA_MODEL = "mistral" 
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text" 

# --- Functions for RAG Process ---

@st.cache_resource
def setup_vector_store(uploaded_file):
    """Processes the uploaded PDF to create a vector store."""
    if uploaded_file is None:
        return None

    # 1. Save uploaded file temporarily
    temp_file_path = f"./temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 2. Load the document
    st.info(f"Processing PDF: {uploaded_file.name} (Using PyMuPDFLoader)...")
    # *** Change 2: Use PyMuPDFLoader ***
    loader = PyMuPDFLoader(temp_file_path)
    docs = loader.load()
    
    # 3. Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # 4. Create embeddings and vector store
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )
    
    # 5. Clean up temporary file
    os.remove(temp_file_path)
    
    st.success("PDF processed and vector store created!")
    return vectorstore.as_retriever()

def generate_response(query, retriever):
    """Creates the RAG chain and generates a response."""
    llm = Ollama(model=OLLAMA_MODEL)
    
    # Create the RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        return_source_documents=False
    )
    
    # Invoke the chain
    response = qa_chain.invoke(query)
    return response['result']

# --- Streamlit UI ---

st.set_page_config(page_title="Local RAG Chatbot (Ollama, LangChain, Streamlit)", layout="wide")
st.title("📄 Local PDF Chatbot (RAG)")
st.caption("Powered by LangChain, Ollama, and Streamlit")

# Sidebar for file upload and settings
with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file and "retriever" not in st.session_state:
        st.session_state["retriever"] = setup_vector_store(uploaded_file)
    
    st.header("2. Models")
    st.info(f"LLM Model: **{OLLAMA_MODEL}**")
    st.info(f"Embedding Model: **{OLLAMA_EMBEDDING_MODEL}**")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main chat input
if prompt := st.chat_input("Ask a question about the document..."):
    # 1. Check if a document has been processed
    if "retriever" not in st.session_state:
        st.warning("Please upload a PDF file first to start the RAG process.")
        
    else:
        # 2. Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 3. Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Pass the prompt to the RAG function
                    response = generate_response(prompt, st.session_state["retriever"])
                    st.markdown(response)
                    # 4. Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_message = f"An error occurred: {e}. Is your Ollama server running and are the models pulled?"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})