from app.langchain.database import ArcadeDBClient
from app.langchain.ingestion import IngestionPipeline
from app.langchain.retrieval import Retriever

def test_search():
    print("Testing search with gemma:2b...")
    db = ArcadeDBClient()
    # Explicitly use "ollama" which is now configured to use "gemma:2b"
    pipeline = IngestionPipeline(db, "ollama") 
    retriever = Retriever(db, pipeline)
    
    results = retriever.search("tensor", limit=3)
    print(f"Found {len(results)} results.")
    for r in results:
        print(r[:100] + "...")

if __name__ == "__main__":
    test_search()
