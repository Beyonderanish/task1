import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np

# --- Configuration ---
LLM_MODEL = "llama3"
EMBEDDING_MODEL = "llama3" # Use the same model for embeddings for simplicity with Ollama
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_CHUNKS = 3

# --- Helper Functions ---

def load_and_process_pdf(pdf_file):
    """
    Loads a PDF, extracts text, chunks it, and generates embeddings using Ollama.
    Stores the chunks and embeddings in Streamlit session state.
    """
    st.session_state['status_message'] = "Extracting text from PDF..."
    st.session_state['document_loaded'] = False

    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() if page.extract_text() else ""

        # Chunk the text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # Initialize Ollama Embeddings (FIXED: Ensures we use Ollama, not GoogleGenerativeAIEmbeddings)
        embeddings_model = OllamaEmbeddings(model=EMBEDDING_MODEL)

        st.session_state['status_message'] = f"Generating embeddings for {len(chunks)} chunks using {EMBEDDING_MODEL}..."
        
        # Generate embeddings
        embeddings = embeddings_model.embed_documents(chunks)
        
        # Store results
        st.session_state['chunks'] = chunks
        st.session_state['embeddings'] = embeddings
        st.session_state['document_loaded'] = True
        st.session_state['document_title'] = pdf_file.name
        st.session_state['status_message'] = f"✅ Document loaded: **{pdf_file.name}**. Ready to chat!"
        st.success(f"Successfully processed {len(chunks)} text chunks.")

    except Exception as e:
        # Catch and display errors, especially if Ollama isn't running
        st.error(f"Error processing PDF or generating embeddings. Ensure Ollama is running and '{LLM_MODEL}' is pulled. Error: {e}")
        st.session_state['status_message'] = f"❌ Error: {e}"
        st.session_state['document_loaded'] = False

def get_most_relevant_context(query_embedding):
    """
    Performs cosine similarity search to retrieve the top K relevant text chunks.
    """
    chunks = st.session_state['chunks']
    embeddings = st.session_state['embeddings']
    
    # Convert query embedding to NumPy array
    query_vector = np.array(query_embedding)
    
    similarities = []
    
    for i, doc_embedding in enumerate(embeddings):
        doc_vector = np.array(doc_embedding)
        
        # Calculate Cosine Similarity
        dot_product = np.dot(query_vector, doc_vector)
        norm_q = np.linalg.norm(query_vector)
        norm_d = np.linalg.norm(doc_vector)
        
        similarity = dot_product / (norm_q * norm_d) if norm_q * norm_d else 0
        
        similarities.append((similarity, chunks[i]))

    # Sort by similarity score (descending)
    similarities.sort(key=lambda x: x[0], reverse=True)
    
    # Filter for chunks with a reasonable score and take the top K
    relevant_chunks = [text for score, text in similarities if score > 0.6][:TOP_K_CHUNKS]
    
    return relevant_chunks

def handle_user_input():
    """
    Handles the user query and generates a response using RAG or general chat.
    """
    # Use key "prompt" for the input
    user_query = st.session_state.get('prompt', '').strip()

    if user_query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Determine if we use RAG context
        context = ""
        source_note = "Answering with **general knowledge**."
        prompt_template = user_query # Default prompt

        if st.session_state.get('document_loaded', False):
            try:
                # 1. Embed the user query
                embeddings_model = OllamaEmbeddings(model=EMBEDDING_MODEL)
                # NOTE: This single call is safe and doesn't require Google credentials
                query_embedding = embeddings_model.embed_query(user_query) 
                
                # 2. Retrieve context
                relevant_chunks = get_most_relevant_context(query_embedding)

                if relevant_chunks:
                    context = "\n---\n".join(relevant_chunks)
                    source_note = f"Answering based on content from **{st.session_state['document_title']}**."
                    
                    # 3. Construct the RAG Prompt
                    system_prompt = (
                        "You are a specialized document Q&A assistant. "
                        "Use ONLY the provided context below to answer the user's question. "
                        "If the answer is not in the context, clearly state that you cannot find the information in the document."
                    )
                    prompt_template = f"{system_prompt}\n\nContext:\n---\n{context}\n---\n\nUser Question: \"{user_query}\"\n\nDetailed Answer:"
                else:
                    # Context not relevant, use general prompt
                    source_note = "No highly relevant context found in the loaded document. Answering with **general knowledge**."

            except Exception as e:
                st.error(f"Error during RAG process. Using general chat mode. Error: {e}")
                prompt_template = user_query
                source_note = "RAG failed. Answering with **general knowledge**."
        
        # 4. Call Ollama (Llama 3)
        with st.spinner("AI is thinking..."):
            llm = Ollama(model=LLM_MODEL)
            response = llm.invoke(prompt_template)
            
            # Combine response and source note
            full_response = f"{response}\n\n---\n\n<small>*Source:* {source_note}</small>"

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response, "is_markdown": True})
        # Note: Streamlit automatically clears the input field on submit


# --- Streamlit UI Setup ---

st.set_page_config(page_title="Ollama Llama 3 RAG Chatbot (Streamlit)", layout="wide")
st.title("📄 Llama 3 PDF Chatbot with Ollama & Streamlit")
st.caption("This RAG app uses Ollama for both Llama 3 generation and embedding generation.")

# Initialize session state for chat
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Please upload a PDF to begin document analysis, or start chatting for general knowledge."}]
if 'document_loaded' not in st.session_state:
    st.session_state['document_loaded'] = False
if 'status_message' not in st.session_state:
    st.session_state['status_message'] = "Waiting for PDF upload..."

# --- Sidebar for PDF Upload ---
with st.sidebar:
    st.header("1. Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", on_change=lambda: st.session_state.update(prompt=""))
    
    # Check if a new file was uploaded or if the file changed
    if uploaded_file and (not st.session_state.get('document_title') or st.session_state.get('document_title') != uploaded_file.name):
        load_and_process_pdf(uploaded_file)
    
    # Display document status
    st.markdown(f"**Document Status:**")
    st.markdown(st.session_state['status_message'])

    st.divider()
    st.header("2. Environment Setup")
    st.markdown("Ensure you have **Ollama** running locally and the **Llama 3** model pulled.")
    st.code(f"ollama pull {LLM_MODEL}", language="bash")
    st.caption("If you encounter connection errors, check your Ollama service status.")


# --- Main Chat Interface ---

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Use markdown for the full response to handle the source note formatting
        st.markdown(message["content"], unsafe_allow_html=True) 

# Chat input at the bottom
# Input is always enabled for general conversation
st.chat_input("Ask a question...", on_submit=handle_user_input, key="prompt")
