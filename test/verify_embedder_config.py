from app.langchain.ingestion import IngestionPipeline
from app.langchain.database import ArcadeDBClient

def verify_config():
    print("Initializing IngestionPipeline with provider='ollama'...")
    try:
        db = ArcadeDBClient()
        # Mock connection or just use it
        pipeline = IngestionPipeline(db, "ollama")
        model = pipeline.embedder.model
        print(f"VERIFIED: The embedding model for 'ollama' provider is: '{model}'")
        
        if model == "gemma:2b":
            print("CONFIRMED: Using Gemma for embeddings.")
        elif model == "nomic-embed-text":
            print("WARNING: Using Nomic for embeddings.")
        else:
            print(f"Using unknown model: {model}")
            
    except Exception as e:
        print(f"Error during verification: {e}")

if __name__ == "__main__":
    verify_config()
