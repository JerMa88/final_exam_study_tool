from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import shutil
import os
from typing import Optional

from .database import ArcadeDBClient
from .ingestion import IngestionPipeline
from .retrieval import Retriever
from .pipeline import RAGPipeline

app = FastAPI()

# dependencies
db_client = ArcadeDBClient()
ingestion = IngestionPipeline(db_client, embedding_provider="vertex") # Default, can toggle
retriever = Retriever(db_client, ingestion) # helper to access embedder
bot = RAGPipeline(retriever, db_client)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, "../../database/PDFs")
os.makedirs(PDF_DIR, exist_ok=True)

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    model: str = "models/gemini-2.0-flash"
    embedding_provider: str = "vertex" # "vertex" or "ollama"
    llm_provider: str = "vertex" # "vertex" or "ollama"

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...), 
    type: str = Form(..., regex="^(slide|textbook)$"),
    embedding_provider: str = Form("vertex")
):
    try:
        file_path = os.path.join(PDF_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Re-init ingestion with correct provider if needed, or just set it
        # ideally we don't recreate pipelines per request but for simplicity:
        temp_ingest = IngestionPipeline(db_client, embedding_provider)
        temp_ingest.process_pdf(file_path, type)
        
        return {"status": "success", "filename": file.filename, "message": "Ingestion started/completed."}
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

@app.delete("/reset")
def reset_db():
    # Dangerous! For dev only.
    # db_client.execute_command("DROP DATABASE ...") 
    pass

@app.get("/health")
def health():
    return {"status": "ok"}
