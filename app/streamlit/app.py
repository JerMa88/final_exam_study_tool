import streamlit as st
import requests
import json

# Backend URL
API_URL = "http://localhost:8000"

st.set_page_config(page_title="Final Exam Study Tool", layout="wide")

# Sidebar Configuration
with st.sidebar:
    st.title("Study Tool Config")
    
    # Provider Selection
    provider = st.radio("LLM Provider", ["vertex", "ollama"], index=0)
    
    # Model Selection
    if provider == "vertex":
        # When using API Key, these map to standard Google GenAI models
        model_options = [
            "models/gemini-2.0-flash",
            "models/gemini-2.5-pro",
            "models/gemma-3-12b-it" 
        ]
    else: # ollama
        model_options = [
            "gemma:2b",
            "gemma:7b",
            "qwen2.5:0.5b" 
        ]
    
    selected_model = st.selectbox("Select Model", model_options)
    
    st.divider()
    
    # File Upload
    st.subheader("Upload Documents")
    uploaded_file = st.file_uploader("Upload PDF or Text", type=["pdf", "txt"])
    doc_type = st.selectbox("Document Type", ["slide", "textbook", "paper"])
    
    if uploaded_file and st.button("Ingest File"):
        with st.spinner("Ingesting..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            data = {"type": doc_type, "embedding_provider": provider} # Use same provider for embedding?
            # Note: Switching embedding provider might require re-indexing or separate spaces.
            # ideally ingestion provider is sticky. For now, passing current provider.
            try:
                res = requests.post(f"{API_URL}/upload", files=files, data=data)
                if res.status_code == 200:
                    st.success(f"Uploaded {uploaded_file.name}")
                else:
                    st.error(f"Upload failed: {res.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")

    st.divider()
    st.subheader("History")
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.rerun()

# Session State Init
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""

# Main Interface
st.title("Final Exam Study Tool 📚")

tab1, tab2 = st.tabs(["Chat 💬", "Search 🔍"])

with tab1:
    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    if prompt := st.chat_input("Ask a question about your materials..."):
        # Add User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Assistant Response Stream
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            debug_info = st.empty() # Placeholder for prompt debug
            
            # Prepare payload
            payload = {
                "message": prompt,
                "conversation_id": st.session_state.conversation_id,
                "model": selected_model,
                "llm_provider": provider,
                "embedding_provider": provider # Matching embedding to LLM provider logic for now
            }
            
            try:
                with requests.post(f"{API_URL}/chat", json=payload, stream=True) as response:
                    if response.status_code == 200:
                        for line in response.iter_lines():
                            if line:
                                try:
                                    data = json.loads(line.decode('utf-8'))
                                    event_type = data.get("type")
                                    
                                    if event_type == "debug":
                                        st.session_state.last_prompt = data.get("prompt")
                                        
                                    elif event_type == "token":
                                        chunk = data.get("content")
                                        full_response += chunk
                                        message_placeholder.markdown(full_response + "▌")
                                        
                                except json.JSONDecodeError:
                                    pass # Skip invalid JSON lines
                        
                        message_placeholder.markdown(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        
                        # Store simple ID if backend actually returned one? 
                        # Current implementation doesn't return ID in stream explicitly, 
                        # so we rely on client-side state or implicit session.
                    else:
                        st.error(f"Error ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

    # Debug View (Outside stream loop, persistent)
    if st.session_state.last_prompt:
        with st.expander("🛠️ Debug: Last Prompt Used"):
            st.code(st.session_state.last_prompt)

with tab2:
    st.header("Direct Semantic Search 🔍")
    search_query = st.text_input("Enter search query", placeholder="What are the key concepts in chapter 1?")
    
    if search_query and st.button("Search"):
         with st.spinner("Searching..."):
            # Update payload for search (using chat request schema or simple dict)
            # Backend expects ChatRequest on POST /search for simplicity
            payload = {
                "message": search_query,
                "embedding_provider": provider
            }
            try:
                res = requests.post(f"{API_URL}/search", json=payload)
                if res.status_code == 200:
                    results = res.json().get("results", [])
                    if results:
                        for i, r in enumerate(results):
                             with st.expander(f"Result {i+1}", expanded=True):
                                 st.markdown(r)
                    else:
                        st.info("No results found.")
                else:
                    st.error(f"Search failed: {res.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")
