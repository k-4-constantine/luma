# frontend/app.py
import os
import streamlit as st
from components.chat_interface import render_chat_interface
from components.document_panel import render_document_panel
from utils.api_client import APIClient

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retrieved_documents" not in st.session_state:
    st.session_state.retrieved_documents = []
if "api_client" not in st.session_state:
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    st.session_state.api_client = APIClient(base_url=backend_url)

st.set_page_config(
    page_title="Luma RAG - Research Assistant",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Luma Research Assistant")

# Two columns: chat (left) and documents (right)
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Chat")
    render_chat_interface(st)

with col2:
    st.subheader("Retrieved Documents")
    render_document_panel(st)