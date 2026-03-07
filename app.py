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
    # Load the local embedding model
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()
embedding_dim = 384
index = faiss.IndexFlatL2(embedding_dim)
tokenizer = tiktoken.get_encoding("cl100k_base")

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    return len(tokenizer.encode(string))

def chunk_text(text, max_tokens=250):
    """Splits text into smaller chunks based on token count."""
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

# --- NEW CONTENT STARTING HERE ---

def extract_text_from_pdf(pdf_file):
    """Extracts raw text from an uploaded PDF file."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

