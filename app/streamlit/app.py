import streamlit as st
import requests
import json
from urllib.parse import quote

# Backend URL
API_URL = "http://localhost:8000"

st.set_page_config(page_title="Final Exam Study Tool", layout="wide")

# ─── Helper Functions ────────────────────────────────────────────────

def encode_rid(rid: str) -> str:
    """URL-encode an ArcadeDB RID (e.g. '#30:0' -> '%2330:0')."""
    if rid and rid.startswith("#"):
        return quote(rid, safe=":")
    return rid

def api_get(path):
    """GET request to backend."""
    try:
        res = requests.get(f"{API_URL}{path}", timeout=10)
        if res.status_code == 200:
            return res.json()
        else:
            st.error(f"API Error ({res.status_code}): {res.text[:200]}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Is the server running on port 8000?")
    except Exception as e:
        st.error(f"Connection error: {e}")
    return None

def api_post(path, json_data=None):
    """POST request to backend."""
    try:
        res = requests.post(f"{API_URL}{path}", json=json_data, timeout=30)
        if res.status_code == 200:
            return res.json()
        else:
            st.error(f"API Error ({res.status_code}): {res.text[:200]}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Is the server running on port 8000?")
    except Exception as e:
        st.error(f"Connection error: {e}")
    return None

def api_delete(path):
    """DELETE request to backend."""
    try:
        res = requests.delete(f"{API_URL}{path}", timeout=10)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        st.error(f"Connection error: {e}")
    return None

def api_put(path, json_data=None):
    """PUT request to backend."""
    try:
        res = requests.put(f"{API_URL}{path}", json=json_data, timeout=10)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        st.error(f"Connection error: {e}")
    return None

def load_session_data(session_rid):
    """Load full session data (files, messages, conversation_id)."""
    data = api_get(f"/sessions/{encode_rid(session_rid)}")
    if data and "session" in data:
        st.session_state.current_session_rid = session_rid
        st.session_state.current_session_name = data["session"].get("name", "Untitled")
        st.session_state.conversation_id = data.get("conversation_id") or None
        # Load chat history from DB
        st.session_state.messages = [
            {"role": m["role"], "content": m["content"]}
            for m in data.get("messages", [])
        ]
        st.session_state.session_files = data.get("files", [])
        return data
    return None

# ─── Session State Init ─────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "current_session_rid" not in st.session_state:
    st.session_state.current_session_rid = None
if "current_session_name" not in st.session_state:
    st.session_state.current_session_name = None
if "session_files" not in st.session_state:
    st.session_state.session_files = []
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""
if "sessions_list" not in st.session_state:
    st.session_state.sessions_list = []

# ─── Sidebar ─────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📚 Study Tool")
    
    # --- Provider & Model Selection ---
    provider = st.radio("LLM Provider", ["vertex", "ollama"], index=1, horizontal=True)
    
    if provider == "vertex":
        model_options = [
            "models/gemini-2.0-flash",
            "models/gemini-2.5-flash",
            "models/gemini-2.5-pro",
        ]
    else:
        model_options = [
            "gemma4:latest",
            "gemma4:26b",
            "gemma3:270m",
        ]
    
    selected_model = st.selectbox("Select Model", model_options)
    
    st.divider()
    
    # --- Study Sessions ---
    st.subheader("📂 Study Sessions")
    
    # Fetch sessions list
    sessions_data = api_get("/sessions")
    if sessions_data:
        st.session_state.sessions_list = sessions_data.get("sessions", [])
    
    # Create new session
    with st.expander("➕ New Session", expanded=not st.session_state.current_session_rid):
        new_session_name = st.text_input(
            "Session Name", 
            placeholder="e.g., Quantum Midterm Prep",
            key="new_session_input"
        )
        if st.button("Create Session", use_container_width=True) and new_session_name:
            result = api_post("/sessions", {"name": new_session_name})
            if result and result.get("rid"):
                load_session_data(result["rid"])
                st.rerun()
    
    # Session list
    if st.session_state.sessions_list:
        for session in st.session_state.sessions_list:
            s_rid = session.get("@rid", "")
            s_name = session.get("name", "Untitled")
            is_active = s_rid == st.session_state.current_session_rid
            
            col1, col2 = st.columns([5, 1])
            with col1:
                label = f"{'▶ ' if is_active else ''}{s_name}"
                if st.button(
                    label, 
                    key=f"session_{s_rid}", 
                    use_container_width=True, 
                    type="primary" if is_active else "secondary"
                ):
                    if not is_active:
                        load_session_data(s_rid)
                        st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{s_rid}", help="Delete session"):
                    api_delete(f"/sessions/{encode_rid(s_rid)}")
                    if s_rid == st.session_state.current_session_rid:
                        st.session_state.current_session_rid = None
                        st.session_state.current_session_name = None
                        st.session_state.messages = []
                        st.session_state.conversation_id = None
                        st.session_state.session_files = []
                    st.rerun()
    else:
        st.caption("No sessions yet. Create one above.")
    
    st.divider()
    
    # --- File Upload (scoped to current session) ---
    st.subheader("📄 Upload Documents")
    
    if not st.session_state.current_session_rid:
        st.info("Create or select a session first.")
    else:
        uploaded_file = st.file_uploader("Upload PDF or Text", type=["pdf", "txt"], key="file_uploader")
        doc_type = st.selectbox("Document Type", ["slide", "textbook", "paper"])
        
        if uploaded_file and st.button("Ingest File", use_container_width=True):
            with st.spinner("Ingesting & embedding..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                data = {
                    "type": doc_type, 
                    "embedding_provider": "ollama",
                    "session_id": st.session_state.current_session_rid,
                }
                try:
                    res = requests.post(f"{API_URL}/upload", files=files, data=data)
                    if res.status_code == 200:
                        st.success(f"✅ Uploaded & linked: {uploaded_file.name}")
                        load_session_data(st.session_state.current_session_rid)
                        st.rerun()
                    else:
                        st.error(f"Upload failed: {res.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")
        
        # Link existing documents to session
        with st.expander("📎 Link Existing Documents"):
            all_docs = api_get("/documents")
            if all_docs:
                docs = all_docs.get("documents", [])
                session_file_rids = {f.get("@rid", "") for f in st.session_state.session_files}
                
                unlinked = [d for d in docs if d.get("@rid", "") not in session_file_rids]
                if unlinked:
                    for doc in unlinked:
                        d_rid = doc.get("@rid", "")
                        d_name = doc.get("filename", "?")
                        d_type = doc.get("type", "?")
                        if st.button(
                            f"📎 {d_name} ({d_type})", 
                            key=f"link_{d_rid}", 
                            use_container_width=True
                        ):
                            api_post(
                                f"/sessions/{encode_rid(st.session_state.current_session_rid)}/files", 
                                {"file_rid": d_rid}
                            )
                            load_session_data(st.session_state.current_session_rid)
                            st.rerun()
                else:
                    st.caption("All documents are already linked.")
    
    st.divider()
    
    # --- Session actions ---
    if st.session_state.current_session_rid:
        if st.button("🔄 New Conversation", use_container_width=True, 
                     help="Start a fresh conversation in this session"):
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.rerun()
        
        if st.button("📤 Leave Session", use_container_width=True, type="secondary"):
            st.session_state.current_session_rid = None
            st.session_state.current_session_name = None
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.session_state.session_files = []
            st.rerun()

# ─── Main Interface ──────────────────────────────────────────────────

# Header with session context
if st.session_state.current_session_rid:
    st.title(f"📚 {st.session_state.current_session_name}")
    
    # Show linked files as pills
    if st.session_state.session_files:
        file_names = [f.get("filename", "?") for f in st.session_state.session_files]
        st.caption(f"📄 **Materials ({len(file_names)}):** {' • '.join(file_names)}")
    else:
        st.warning("⚠️ No files linked. Upload or link documents in the sidebar to enable RAG.")
else:
    st.title("Final Exam Study Tool 📚")
    st.info("👈 Create or select a **Study Session** in the sidebar to get started.")

tab1, tab2 = st.tabs(["Chat 💬", "Search 🔍"])

with tab1:
    if not st.session_state.current_session_rid:
        st.info("Select a study session to start chatting.")
    else:
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
                
                # Prepare payload
                payload = {
                    "message": prompt,
                    "conversation_id": st.session_state.conversation_id,
                    "session_id": st.session_state.current_session_rid,
                    "model": selected_model,
                    "llm_provider": provider,
                    "embedding_provider": "ollama"
                }
                
                try:
                    with requests.post(f"{API_URL}/chat", json=payload, stream=True, timeout=300) as response:
                        if response.status_code == 200:
                            for line in response.iter_lines():
                                if line:
                                    try:
                                        data = json.loads(line.decode('utf-8'))
                                        event_type = data.get("type")
                                        
                                        if event_type == "meta":
                                            # Track conversation ID from backend
                                            st.session_state.conversation_id = data.get("conversation_id")
                                        
                                        elif event_type == "debug":
                                            st.session_state.last_prompt = data.get("prompt")
                                            
                                        elif event_type == "token":
                                            chunk = data.get("content")
                                            full_response += chunk
                                            message_placeholder.markdown(full_response + "▌")
                                            
                                    except json.JSONDecodeError:
                                        pass
                            
                            message_placeholder.markdown(full_response)
                            st.session_state.messages.append({"role": "assistant", "content": full_response})
                        else:
                            st.error(f"Error ({response.status_code}): {response.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")

        # Debug View
        if st.session_state.last_prompt:
            with st.expander("🛠️ Debug: Last Prompt Used"):
                st.code(st.session_state.last_prompt)

with tab2:
    st.header("Direct Semantic Search 🔍")
    
    if st.session_state.current_session_rid:
        st.caption(f"Searching within: **{st.session_state.current_session_name}**")
    else:
        st.caption("Searching all documents (no session selected)")
    
    search_query = st.text_input("Enter search query", placeholder="What are the key concepts in chapter 1?")
    
    if search_query and st.button("Search"):
         with st.spinner("Searching..."):
            payload = {
                "message": search_query,
                "embedding_provider": "ollama",
                "session_id": st.session_state.current_session_rid,
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
