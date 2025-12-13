from langchain_community.embeddings import OllamaEmbeddings

def test_gemma_embedding():
    print("Testing gemma:2b as embedder...")
    try:
        # gemma:2b is a chat model, but Ollama might allow embedding with it
        embedder = OllamaEmbeddings(model="gemma:2b")
        vec = embedder.embed_query("Hello world")
        print(f"Success! Vector length: {len(vec)}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_gemma_embedding()
