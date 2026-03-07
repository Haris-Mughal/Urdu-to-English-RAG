import os
from dotenv import load_dotenv
import fitz 
import streamlit as st
import tempfile
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import tiktoken
import requests
from deep_translator import GoogleTranslator
from gtts import gTTS
import time

# Load environment variables
load_dotenv()

# Basic Streamlit Page Configuration
st.set_page_config(
    page_title="RAG-Explorer-AI-Document-Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize Session State
for key, default in {
    "chunks": [],
    "chunk_sources": [],
    "debug_mode": False,
    "last_query_time": None,
    "last_response": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

def get_api_key():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY environment variable is not set.")
    return api_key

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()
embedding_dim = 384
index = faiss.IndexFlatL2(embedding_dim)
tokenizer = tiktoken.get_encoding("cl100k_base")

def num_tokens_from_string(string: str) -> int:
    return len(tokenizer.encode(string))

def chunk_text(text, max_tokens=250):
    sentences = text.split(". ")
    current_chunk = []
    total_tokens = 0
    result_chunks = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        token_len = num_tokens_from_string(sentence)
        if total_tokens + token_len > max_tokens:
            if current_chunk:
                result_chunks.append(". ".join(current_chunk) + ("." if not current_chunk[-1].endswith(".") else ""))
            current_chunk = [sentence]
            total_tokens = token_len
        else:
            current_chunk.append(sentence)
            total_tokens += token_len
    if current_chunk:
        result_chunks.append(". ".join(current_chunk) + ("." if not current_chunk[-1].endswith(".") else ""))
    return result_chunks

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def index_uploaded_text(text):
    global index
    index = faiss.IndexFlatL2(embedding_dim)
    st.session_state.chunks = []
    st.session_state.chunk_sources = []
    chunks_list = chunk_text(text)
    st.session_state.chunks = chunks_list
    for i, chunk in enumerate(chunks_list):
        st.session_state.chunk_sources.append(f"Chunk {i+1}: {chunk[:50]}...")
        vector = embedder.encode([chunk])[0]
        index.add(np.array([vector]).astype('float32'))
    return len(chunks_list)

def retrieve_chunks(query, top_k=5):
    if index.ntotal == 0:
        return []
    q_vector = embedder.encode([query])
    D, I = index.search(np.array(q_vector).astype('float32'), k=min(top_k, index.ntotal))
    return [st.session_state.chunks[i] for i in I[0] if i < len(st.session_state.chunks)]

def build_prompt(system_prompt, context_chunks, question):
    context = "\n\n".join(context_chunks)
    return f"{system_prompt}\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer: Please provide a comprehensive answer based only on the context provided."

def generate_answer(prompt):
    api_key = get_api_key()
    if not api_key:
        return "API key error."
    headers = {"Authorization": f"Bearer {api_key.strip()}", "Content-Type": "application/json"}
    selected_model = st.session_state.get("MODEL_CHOICE", "llama-3.1-8b-instant")
    payload = {
        "model": selected_model,
        "messages": [
            {"role": "system", "content": "You are a helpful document assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1024
    }
    try:
        start_time = time.time()
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
        st.session_state.last_query_time = f"{time.time() - start_time:.2f} seconds"
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"]
        st.session_state.last_response = answer
        return answer
    except Exception as e:
        return f"Error: {str(e)}"

def translate_text(text, target_language):
    try:
        return GoogleTranslator(source='auto', target=target_language).translate(text)
    except Exception:
        return text

# UI Header
st.title("📄 RAG Explorer: AI-Powered Document Assistant & Translator")
st.markdown("Upload a document and ask questions.")

# Sidebar
with st.sidebar:    
    st.subheader("Model Selection")
    st.session_state["MODEL_CHOICE"] = st.selectbox("Select LLM Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"])
    if st.session_state.last_query_time:
         st.subheader("About")
         st.markdown("This app uses RAG to answer questions about documents.")

col1, col2 = st.columns([2, 1])
with col1:
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
    if uploaded_file:
        raw_text = extract_text_from_pdf(uploaded_file) if uploaded_file.type == "application/pdf" else uploaded_file.read().decode("utf-8")
        index_uploaded_text(raw_text)
        st.success("Indexed successfully!")

# --- NEW CONTENT STARTING HERE ---

st.divider()
query = st.text_input("Ask a question about the document")

c1, c2 = st.columns([1, 1])
with c1:
    use_local = st.checkbox("Use local processing (no API call)", value=False)
with c2:
    language = st.selectbox("Language", ["English", "Urdu"])
    lang_code = "en" if language == "English" else "ur"

if st.button("Get Answer", type="primary") and query:
    if index.ntotal == 0:
        st.warning("Please upload a document first.")
    else:
        with st.spinner("Generating..."):
            top_chunks = retrieve_chunks(query)
            if not top_chunks:
                st.error("No relevant content found.")
            else:
                if use_local:
                    answer = f"Local relevant passages:\n\n" + "\n\n".join(top_chunks[:3])
                else:
                    prompt = build_prompt("Use context only.", top_chunks, query)
                    answer = generate_answer(prompt)
                
                translated = translate_text(answer, lang_code)
                st.markdown(f"### Answer ({language}):")
                st.write(translated)                            

