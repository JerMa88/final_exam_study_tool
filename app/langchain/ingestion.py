import fitz  # pymupdf
import os
from typing import List, Literal, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
# from langchain_ollama import OllamaEmbeddings # Newer import if available, fallback to community
from .database import ArcadeDBClient

class IngestionPipeline:
    def __init__(self, db_client: ArcadeDBClient, embedding_provider: str = "vertex"):
        self.db = db_client
        self.embedder = self._get_embedder(embedding_provider)
    
    def _get_embedder(self, provider: str):
        if provider == "ollama":
            # Assuming 'embedding-gemma' model name or similar
            return OllamaEmbeddings(model="nomic-embed-text") # Or embedding-gemma if pulled
        else:
            # Default to Vertex AI
            return VertexAIEmbeddings(model_name="text-embedding-004")

    def process_pdf(self, file_path: str, doc_type: Literal['slide', 'textbook']):
        filename = os.path.basename(file_path)
        print(f"Processing {filename} as {doc_type}...")
        
        # 1. Register Document
        doc_rid = self.db.insert_document(filename, doc_type)
        if not doc_rid:
            print("Failed to insert document metadata.")
            return

        # 2. Extract and Chunk
        doc = fitz.open(file_path)
        chunks = []
        
        if doc_type == 'slide':
            # One chunk per page
            for i, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    chunks.append({"content": text, "page": i + 1})
        else:
            # Recursive split for textbooks
            full_text = ""
            page_map = [] # character index to page number roughly
            for i, page in enumerate(doc):
                text = page.get_text()
                full_text += text + "\n"
                # This mapping is naive for splitters, but simplified: 
                # passing page numbers in splitters is complex without custom splitters.
                # For now, we will chunk page by page then split chunks if too large, 
                # OR just split the whole text and lose precise page tracking.
                # Better approach for textbooks: Chunk per page, then sub-chunk if needed.
                chunks.append({"content": text, "page": i + 1})
            
            # Post-process chunks if they are too big? 
            # For simplicity, let's Stick to page-based for both but use splitter for large pages
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
