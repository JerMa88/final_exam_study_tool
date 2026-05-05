from langchain_ollama import OllamaEmbeddings

def test_gemma_embedding():
    print("Testing embeddinggemma as embedder...")
    try:
        embedder = OllamaEmbeddings(model="embeddinggemma")
        vec = embedder.embed_query("Hello world")
        print(f"Success! Vector length: {len(vec)}")
        print(f"First 5 values: {vec[:5]}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_gemma_embedding()
