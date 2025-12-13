from typing import List, Dict
from .ingestion import IngestionPipeline
from .database import ArcadeDBClient

class Retriever:
    def __init__(self, db_client: ArcadeDBClient, ingestion_pipeline: IngestionPipeline):
        self.db = db_client
        self.pipeline = ingestion_pipeline # To access the embedder
        
    def search(self, query: str, limit: int = 5) -> List[str]:
        # 1. Embed query
        embedding = self.pipeline.embedder.embed_query(query)
        
        # 2. Search DB
        results = self.db.similarity_search(embedding, limit)
        
        # 3. Format results
        # Assuming results have 'content' and 'distance'
        docs = []
        for r in results:
            content = r.get("content", "")
            page = r.get("page_number", "?")
            # score = r.get("distance", 0) # Use if needed for filtering
            docs.append(f"[Page {page}] {content}")
            
        return docs
