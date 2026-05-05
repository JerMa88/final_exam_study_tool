from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import shutil
import os
import uvicorn
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load env vars
load_dotenv()

from .database import ArcadeDBClient
from .ingestion import IngestionPipeline
from .retrieval import Retriever
from .pipeline import RAGPipeline
from .cheatsheet_pipeline import CheatSheetPipeline

app = FastAPI()

# Database Client
db_client = ArcadeDBClient()

# Initialize Retrieval
# UPDATED: defaulting to "ollama" per user request to use Gemma embeddings
ingestion_pipeline = IngestionPipeline(db_client, "ollama") 
retriever = Retriever(db_client, ingestion_pipeline)
bot = RAGPipeline(retriever, db_client)
cheatsheet_bot = CheatSheetPipeline(retriever, db_client)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, "../../database/PDFs")
os.makedirs(PDF_DIR, exist_ok=True)

@app.on_event("startup")
async def startup_event():
    print("Checking for new files to ingest on startup...")
    # Map subdirs to doc types
    type_map = {
        "slides": "slide",
        "textbooks": "textbook",
        "papers": "paper",
        "texts": "textbook" # Treat texts as textbooks (recursive split)
    }
    
    # Ensure PDF roots exist
    os.makedirs(PDF_DIR, exist_ok=True)
    
    for subdir, doc_type in type_map.items():
        dir_path = os.path.join(PDF_DIR, subdir)
        if not os.path.exists(dir_path):
            continue
            
        for filename in os.listdir(dir_path):
            if filename.lower().endswith(".pdf") or (subdir == "texts" and filename.lower().endswith(".txt")):
                if not db_client.document_exists(filename):
                     print(f"[Startup] Ingesting new file: {filename} ({doc_type})")
                     file_path = os.path.join(dir_path, filename)
                     try:
                        ingestion_pipeline.process_pdf(file_path, doc_type)
                     except Exception as e:
                        print(f"Error ingesting {filename}: {e}")
                else:
                    print(f"[Startup] Skipping existing file: {filename}")
                    pass
    print("Startup check complete.")

# ─── Request Models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None  # Study session RID
    model: str = "models/gemini-2.0-flash"
    embedding_provider: str = "vertex"
    llm_provider: str = "vertex"

class CreateSessionRequest(BaseModel):
    name: str

class RenameSessionRequest(BaseModel):
    name: str

class LinkFileRequest(BaseModel):
    file_rid: str

class GenerateCheatSheetRequest(BaseModel):
    topics: List[str]
    session_id: str
    model: str = "models/gemini-2.0-flash"
    llm_provider: str = "vertex"
    embedding_provider: str = "vertex"

class RefineCheatSheetRequest(BaseModel):
    instruction: str
    session_id: str
    model: str = "models/gemini-2.0-flash"
    llm_provider: str = "vertex"
    embedding_provider: str = "vertex"

# ─── File Upload ─────────────────────────────────────────────────────

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...), 
    type: str = Form(..., pattern="^(slide|textbook|paper)$"),
    embedding_provider: str = Form("ollama"),
    session_id: str = Form(None),
):
    try:
        # Determine subdirectory
        if file.filename.lower().endswith(".txt"):
            subdir = "texts"
        elif type == "slide":
            subdir = "slides"
        elif type == "textbook":
            subdir = "textbooks"
        elif type == "paper":
            subdir = "papers"
        else:
            subdir = ""

        target_dir = os.path.join(PDF_DIR, subdir)
        os.makedirs(target_dir, exist_ok=True)
        
        file_path = os.path.join(target_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Ingest
        temp_ingest = IngestionPipeline(db_client, embedding_provider)
        temp_ingest.process_pdf(file_path, type)
        
        # Get the file's RID
        file_rid = db_client.get_document_rid(file.filename)
        
        # Link to session if provided
        if session_id and file_rid:
            db_client.link_file_to_session(session_id, file_rid)
        
        return {
            "status": "success", 
            "filename": file.filename, 
            "type": type, 
            "file_rid": file_rid,
            "session_id": session_id,
            "message": "Ingestion completed."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── Study Sessions ─────────────────────────────────────────────────

@app.get("/sessions")
def list_sessions():
    """List all study sessions."""
    sessions = db_client.list_sessions()
    return {"sessions": sessions}

@app.post("/sessions")
def create_session(req: CreateSessionRequest):
    """Create a new study session."""
    result = db_client.create_session(req.name)
    if not result:
        raise HTTPException(status_code=500, detail="Failed to create session")
    return result

@app.get("/sessions/{session_rid}")
def get_session(session_rid: str):
    """Get session details including files and messages."""
    session = db_client.get_session(session_rid)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    files = db_client.get_session_files(session_rid)
    messages = db_client.get_session_messages(session_rid)
    conv_rid = db_client.get_session_conversation(session_rid)
    
    return {
        "session": session,
        "files": files,
        "messages": messages,
        "conversation_id": conv_rid,
    }

@app.delete("/sessions/{session_rid}")
def delete_session(session_rid: str):
    """Delete a study session."""
    db_client.delete_session(session_rid)
    return {"status": "deleted"}

@app.put("/sessions/{session_rid}")
def rename_session(session_rid: str, req: RenameSessionRequest):
    """Rename a study session."""
    db_client.rename_session(session_rid, req.name)
    return {"status": "renamed", "name": req.name}

@app.post("/sessions/{session_rid}/files")
def link_file(session_rid: str, req: LinkFileRequest):
    """Link an existing file to a session."""
    db_client.link_file_to_session(session_rid, req.file_rid)
    return {"status": "linked"}

@app.delete("/sessions/{session_rid}/files/{file_rid}")
def unlink_file(session_rid: str, file_rid: str):
    """Unlink a file from a session."""
    db_client.unlink_file_from_session(session_rid, file_rid)
    return {"status": "unlinked"}

@app.get("/sessions/{session_rid}/files")
def get_session_files(session_rid: str):
    """Get all files linked to a session."""
    files = db_client.get_session_files(session_rid)
    return {"files": files}

@app.get("/sessions/{session_rid}/messages")
def get_session_messages(session_rid: str):
    """Get chat history for a session."""
    messages = db_client.get_session_messages(session_rid)
    return {"messages": messages}

# ─── Documents ───────────────────────────────────────────────────────

@app.get("/documents")
def list_documents():
    """List all ingested documents."""
    docs = db_client.get_all_documents()
    return {"documents": docs}

# ─── Chat (Session-Aware) ───────────────────────────────────────────

# Track which embedding provider is currently loaded
_current_embedding_provider = "ollama"

@app.post("/chat")
async def chat(req: ChatRequest):
    global _current_embedding_provider
    # Swap embedding provider if it changed
    if req.embedding_provider != _current_embedding_provider:
        bot.retriever.pipeline = IngestionPipeline(db_client, req.embedding_provider)
        _current_embedding_provider = req.embedding_provider

    # If session-scoped, get the file RIDs for this session
    file_rids = None
    if req.session_id:
        file_rids = db_client.get_session_file_rids(req.session_id)
        # If session has no files, search all (graceful fallback)
        if not file_rids:
            file_rids = None

    return StreamingResponse(
        bot.chat_stream(
            req.message, 
            req.conversation_id, 
            req.model, 
            req.llm_provider,
            session_rid=req.session_id,
            file_rids=file_rids,
        ),
        media_type="text/plain"
    )

# ─── Search (Session-Aware) ─────────────────────────────────────────

@app.post("/search")
async def search(req: ChatRequest):
    # Update retriever's embedder
    bot.retriever.pipeline = IngestionPipeline(db_client, req.embedding_provider)
    
    # Session-scoped search
    file_rids = None
    if req.session_id:
        file_rids = db_client.get_session_file_rids(req.session_id)
        if not file_rids:
            file_rids = None
    
    results = retriever.search(req.message, limit=5, file_rids=file_rids)
    return {"results": results}

# ─── Cheat Sheet ─────────────────────────────────────────────────────

@app.post("/cheatsheet/generate")
async def generate_cheatsheet(req: GenerateCheatSheetRequest):
    global _current_embedding_provider
    if req.embedding_provider != _current_embedding_provider:
        cheatsheet_bot.retriever.pipeline = IngestionPipeline(db_client, req.embedding_provider)
        bot.retriever.pipeline = IngestionPipeline(db_client, req.embedding_provider)
        _current_embedding_provider = req.embedding_provider

    file_rids = None
    if req.session_id:
        file_rids = db_client.get_session_file_rids(req.session_id)
        if not file_rids:
            file_rids = None

    return StreamingResponse(
        cheatsheet_bot.generate_stream(
            topics=req.topics,
            session_rid=req.session_id,
            model_name=req.model,
            llm_provider=req.llm_provider,
            file_rids=file_rids,
        ),
        media_type="text/plain"
    )

@app.post("/cheatsheet/refine")
async def refine_cheatsheet(req: RefineCheatSheetRequest):
    global _current_embedding_provider
    if req.embedding_provider != _current_embedding_provider:
        cheatsheet_bot.retriever.pipeline = IngestionPipeline(db_client, req.embedding_provider)
        bot.retriever.pipeline = IngestionPipeline(db_client, req.embedding_provider)
        _current_embedding_provider = req.embedding_provider

    file_rids = None
    if req.session_id:
        file_rids = db_client.get_session_file_rids(req.session_id)
        if not file_rids:
            file_rids = None

    return StreamingResponse(
        cheatsheet_bot.refine_stream(
            instruction=req.instruction,
            session_rid=req.session_id,
            model_name=req.model,
            llm_provider=req.llm_provider,
            file_rids=file_rids,
        ),
        media_type="text/plain"
    )

@app.get("/sessions/{session_rid}/cheatsheet")
def get_session_cheatsheet(session_rid: str):
    """Get the cheat sheet for a session."""
    cs = db_client.get_session_cheatsheet(session_rid)
    return {"cheatsheet": cs}

@app.delete("/sessions/{session_rid}/cheatsheet")
def delete_session_cheatsheet(session_rid: str):
    """Delete the cheat sheet for a session."""
    db_client.delete_session_cheatsheet(session_rid)
    return {"status": "deleted"}

# ─── Utility ─────────────────────────────────────────────────────────

@app.delete("/reset")
def reset_db():
    # Dangerous! For dev only.
    pass

@app.get("/health")
def health():
    return {"status": "ok"}
