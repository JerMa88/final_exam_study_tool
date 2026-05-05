from typing import List, Dict, Optional
from .ingestion import IngestionPipeline
from .database import ArcadeDBClient

class Retriever:
    def __init__(self, db_client: ArcadeDBClient, ingestion_pipeline: IngestionPipeline):
        self.db = db_client
        self.pipeline = ingestion_pipeline  # To access the embedder
        
    def search(self, query: str, limit: int = 5, 
               file_rids: List[str] = None) -> List[str]:
        """
        Search for relevant chunks.
        If file_rids is provided, only search chunks from those files (session-scoped).
        """
        # 1. Embed query
        embedding = self.pipeline.embedder.embed_query(query)
        
        # 2. Search DB (optionally scoped to specific files)
        results = self.db.similarity_search(embedding, limit, file_rids=file_rids)
        
        # 3. Format results
        docs = []
        for r in results:
            content = r.get("content", "")
            page = r.get("page_number", "?")
            docs.append(f"[Page {page}] {content}")
            
        return docs
