import sys
import os
import random

# Ensure app modules are in path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.langchain.database import ArcadeDBClient

def test_manual_search():
    client = ArcadeDBClient(db_name="study_tool")
    
    # Check if we have chunks
    chunks_count = client.query("SELECT count(*) FROM Chunk")[0]["count(*)"]
    print(f"Total chunks: {chunks_count}")
    if chunks_count == 0:
        print("Skipping search test (no chunks).")
        return

    # Create a dummy embedding (768 dim)
    # Using random values to ensure dot product isn't zero
    embedding = [random.random() for _ in range(768)]
    
    print("Running similarity search...")
    results = client.similarity_search(embedding, limit=3)
    
    print(f"Found {len(results)} results.")
    for i, res in enumerate(results):
        print(f"Result {i+1}: Score={res['score']:.4f}, Page={res.get('page_number')}")
        assert "score" in res
        assert "content" in res

    assert len(results) > 0
    print("Search test passed.")

if __name__ == "__main__":
    test_manual_search()
