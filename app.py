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

# Load environment variables (like API keys)
load_dotenv()

# Basic Streamlit Page Configuration
st.set_page_config(
    page_title="RAG-Explorer-AI-Document-Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize Session State for storing data across reruns
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
        st.error("GROQ_API_KEY environment variable is not set. Please set it before running the application.")
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
            total_tokens = total_tokens
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
    return f"""{system_prompt}
Context:
{context}
Question:
{question}
Answer: Please provide a comprehensive answer based only on the context provided."""

def generate_answer(prompt):
    api_key = get_api_key()
    if not api_key:
        return "API key is missing. Please set the GROQ_API_KEY environment variable or enter it in the sidebar."
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json"
    }
    selected_model = st.session_state.get("MODEL_CHOICE", "llama-3.1-8b-instant")
    payload = {
        "model": selected_model,
        "messages": [
            {"role": "system", "content": "You are a helpful document assistant that answers questions only using the provided context."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1024
    }
    try:
        start_time = time.time()
        with st.spinner("Sending request to Groq API..."):
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=30
            )
        query_time = time.time() - start_time
        st.session_state.last_query_time = f"{query_time:.2f} seconds"
        response.raise_for_status()
        response_json = response.json()
        answer = response_json["choices"][0]["message"]["content"]
        st.session_state.last_response = answer
        return answer
    except Exception as e:
        return f"Unexpected error: {str(e)}"

# --- NEW CONTENT STARTING HERE ---

def translate_text(text, target_language):
    """Translates text using Google Translator."""
    try:
        with st.spinner(f"Translating to {target_language}..."):
            return GoogleTranslator(source='auto', target=target_language).translate(text)
    except Exception as e:
        st.error(f"Translation failed: {str(e)}")
        return text

# Streamlit UI Header
st.title("📄 RAG Explorer: AI-Powered Document Assistant & Translator")
st.markdown("Upload a document and ask questions to get AI-powered answers with translation capabilities.")

