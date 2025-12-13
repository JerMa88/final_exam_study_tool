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

app = FastAPI()

# Database Client
db_client = ArcadeDBClient()
db_client._ensure_db_exists()

# Initialize Retrieval
# We need to pick a default embedder for the global retriever or make it dynamic per request.
# For now, initializing with default Vertex (or Ollama if fallback kicks in)
# The pipeline default is Vertex.
# UPDATED: defaulting to "ollama" per user request to use Gemma embeddings
ingestion_pipeline = IngestionPipeline(db_client, "ollama") 
retriever = Retriever(db_client, ingestion_pipeline)
bot = RAGPipeline(retriever, db_client)

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
                     # Use default embedder or smart detection? using default 'vertex' (handled by ingestion pipeline fallback)
                     file_path = os.path.join(dir_path, filename)
                     try:
                        ingestion_pipeline.process_pdf(file_path, doc_type)
                     except Exception as e:
                        print(f"Error ingesting {filename}: {e}")
                else:
                    print(f"[Startup] Skipping existing file: {filename}")
                    pass
    print("Startup check complete.")

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    model: str = "models/gemini-2.0-flash"
    embedding_provider: str = "vertex" # "vertex" or "ollama"
    llm_provider: str = "vertex" # "vertex" or "ollama"

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...), 
    type: str = Form(..., regex="^(slide|textbook|paper)$"),
    embedding_provider: str = Form("vertex")
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
            subdir = "" # Should be covered by regex/logic but safe fallback

        target_dir = os.path.join(PDF_DIR, subdir)
        os.makedirs(target_dir, exist_ok=True)
        
        file_path = os.path.join(target_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Re-init ingestion with correct provider if needed, or just set it
        # ideally we don't recreate pipelines per request but for simplicity:
        temp_ingest = IngestionPipeline(db_client, embedding_provider)
        temp_ingest.process_pdf(file_path, type)
        
        return {"status": "success", "filename": file.filename, "type": type, "path": file_path, "message": "Ingestion started/completed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(req: ChatRequest):
    # Determine embedder for this chat (should match ingestion? 
    # Realistically retrieval uses the same embedder as what was indexed.
    # For now assuming one global index type or hybrid.)
    # Update retriever's embedder if needed
    if req.embedding_provider != bot.retriever.pipeline.embedder.__class__.__name__:
        # Simple hack: swap pipeline in retriever
        bot.retriever.pipeline = IngestionPipeline(db_client, req.embedding_provider)

    return StreamingResponse(
        bot.chat_stream(req.message, req.conversation_id, req.model, req.llm_provider),
        media_type="text/plain"
    )

@app.post("/search")
async def search(req: ChatRequest):
    # Re-use ChatRequest structure for simplicity (message = query)
    # Update retriever's embedder if needed
    # Update retriever's embedder always to ensure correct model usage
    bot.retriever.pipeline = IngestionPipeline(db_client, req.embedding_provider)
    
    results = retriever.search(req.message, limit=5)
    return {"results": results}

@app.delete("/reset")
def reset_db():
    # Dangerous! For dev only.
    # db_client.execute_command("DROP DATABASE ...") 
    pass

@app.get("/health")
def health():
    return {"status": "ok"}
