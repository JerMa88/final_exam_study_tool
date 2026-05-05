import fitz  # pymupdf
import os
from typing import List, Literal, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from .database import ArcadeDBClient

class IngestionPipeline:
    def __init__(self, db_client: ArcadeDBClient, embedding_provider: str = "vertex"):
        self.db = db_client
        self.embedder = self._get_embedder(embedding_provider)
    
    def _get_embedder(self, provider: str):
        try:
            if provider == "ollama":
                from .utils import ensure_ollama_model
                ensure_ollama_model("embeddinggemma")
                return OllamaEmbeddings(model="embeddinggemma")
            else:
                # Check for API Key
                if os.getenv("GOOGLE_API_KEY"):
                    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
                # Default to Vertex AI
                return VertexAIEmbeddings(model_name="gemini-embedding-001")
        except Exception as e:
            print(f"Warning: Failed to initialize embedder '{provider}': {e}")
            raise e

    def process_pdf(self, file_path: str, doc_type: Literal['slide', 'textbook', 'paper']):
        filename = os.path.basename(file_path)
        print(f"Processing {filename} as {doc_type}...")
        
        # 1. Register Document
        doc_rid = self.db.insert_document(filename, doc_type)
        if not doc_rid:
            print("Failed to insert document metadata.")
            return

        # 2. Extract and Chunk
        chunks = []
        
        if file_path.lower().endswith(".txt"):
            # Text file processing
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                # Treat whole text file as potentially one large content to be split
                chunks.append({"content": text, "page": 1})
            except Exception as e:
                print(f"Error reading text file: {e}")
                return
        else:
            # PDF processing
            doc = fitz.open(file_path)
            if doc_type == 'slide':
                # One chunk per page
                for i, page in enumerate(doc):
                    text = page.get_text()
                    if text.strip():
                        chunks.append({"content": text, "page": i + 1})
            else:
                # Recursive split for textbooks and papers
                # For papers, we might want different logic, but treating like textbook is safe default
                for i, page in enumerate(doc):
                    text = page.get_text()
                    if text.strip():
                        chunks.append({"content": text, "page": i + 1})
        
        # Split chunks if needed (for textbooks, papers, and text files)
        if doc_type in ['textbook', 'paper'] or file_path.lower().endswith(".txt"):
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            final_chunks = []
            for c in chunks:
                splits = splitter.split_text(c["content"])
                for s in splits:
                    final_chunks.append({"content": s, "page": c["page"]})
            chunks = final_chunks
        
        # 3. Embed and Insert
        print(f"Embedding {len(chunks)} chunks...")
        for chunk in chunks:
            vector = self.embedder.embed_query(chunk["content"])
            self.db.insert_chunk(doc_rid, chunk["content"], chunk["page"], vector)
            
        print(f"Finished processing {filename}.")

if __name__ == "__main__":
    # Test run
    db = ArcadeDBClient()
    pipeline = IngestionPipeline(db, "ollama")
    # pipeline.process_pdf("test.pdf", "textbook")
