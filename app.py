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

load_dotenv()

st.set_page_config(
    page_title="RAG-Explorer-AI-Document-Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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