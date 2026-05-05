"""
End-to-End Test: Local Ollama Models
Tests embedding via embeddinggemma and chat via gemma4:26b (and gemma4:latest fallback).
Requires: Ollama running with embeddinggemma and gemma4 models pulled.
"""
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

def test_ollama_running():
    """Check that Ollama server is reachable."""
    import requests
    try:
        res = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in res.json().get("models", [])]
        print(f"✅ Ollama server running. Available models: {models}")
        return models
    except Exception as e:
        print(f"❌ Ollama not reachable: {e}")
        return []

def test_local_embedding():
    """Test embeddinggemma via Ollama."""
    from langchain_ollama import OllamaEmbeddings
    
    print("\n--- Testing Local Embedding (embeddinggemma) ---")
    start = time.time()
    embedder = OllamaEmbeddings(model="embeddinggemma")
    vec = embedder.embed_query("What is quantum entanglement?")
    elapsed = time.time() - start
    
    assert isinstance(vec, list), f"Expected list, got {type(vec)}"
    assert len(vec) > 0, "Empty embedding vector"
    print(f"✅ Embedding success: dim={len(vec)}, time={elapsed:.2f}s")
    print(f"   First 5 values: {vec[:5]}")
    return vec

def test_local_chat_gemma4_small():
    """Test gemma4:latest (8B Q4_K_M) — the small/fast model."""
    from langchain_ollama import ChatOllama
    
    print("\n--- Testing Local Chat (gemma4:latest — 8B) ---")
    start = time.time()
    llm = ChatOllama(model="gemma4:latest")
    res = llm.invoke("What is 2+2? Reply with just the number.")
    elapsed = time.time() - start
    
    assert res.content, "Empty response"
    print(f"✅ Chat success: response='{res.content.strip()}', time={elapsed:.2f}s")
    return res.content

def test_local_chat_gemma4_26b():
    """Test gemma4:26b (26B-A4B quantized) — the large model."""
    from langchain_ollama import ChatOllama
    
    print("\n--- Testing Local Chat (gemma4:26b — 26B A4B) ---")
    start = time.time()
    llm = ChatOllama(model="gemma4:26b")
    res = llm.invoke("What is 2+2? Reply with just the number.")
    elapsed = time.time() - start
    
    assert res.content, "Empty response"
    print(f"✅ Chat success: response='{res.content.strip()}', time={elapsed:.2f}s")
    return res.content

def test_local_rag_pipeline():
    """Test the full RAG pipeline with local Ollama models (no DB required)."""
    from langchain_ollama import OllamaEmbeddings, ChatOllama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    print("\n--- Testing Local RAG Pipeline (embeddinggemma + gemma4:26b) ---")
    
    # 1. Embed a query
    embedder = OllamaEmbeddings(model="embeddinggemma")
    query = "Explain Grover's algorithm"
    query_vec = embedder.embed_query(query)
    print(f"   Query embedded: dim={len(query_vec)}")
    
    # 2. Mock context (simulating retrieval)
    mock_context = """
    [Page 5] Grover's algorithm is a quantum algorithm for searching an unsorted database 
    with N entries in O(√N) time, providing a quadratic speedup over classical algorithms.
    It uses amplitude amplification to increase the probability of measuring the correct answer.
    """
    
    # 3. Generate response with gemma4:26b
    template = """Answer the question using the provided context.
    Context: {context}
    Question: {question}
    Response:"""
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOllama(model="gemma4:26b")
    
    chain = (
        {"context": lambda x: mock_context, "question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    start = time.time()
    response = chain.invoke(query)
    elapsed = time.time() - start
    
    assert len(response) > 10, f"Response too short: {response}"
    print(f"✅ RAG pipeline success: time={elapsed:.2f}s")
    print(f"   Response preview: {response[:300]}...")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("E2E TEST: Local Ollama Models")
    print("=" * 60)
    
    results = {}
    
    # Check Ollama is running
    models = test_ollama_running()
    if not models:
        print("Aborting — Ollama server not running.")
        sys.exit(1)
    results["ollama_server"] = True
    
    try:
        results["embedding_gemma"] = test_local_embedding() is not None
    except Exception as e:
        print(f"❌ Local embedding failed: {e}")
        results["embedding_gemma"] = False
    
    try:
        results["chat_gemma4_8b"] = test_local_chat_gemma4_small() is not None
    except Exception as e:
        print(f"❌ Local chat (8B) failed: {e}")
        results["chat_gemma4_8b"] = False
    
    try:
        results["chat_gemma4_26b"] = test_local_chat_gemma4_26b() is not None
    except Exception as e:
        print(f"❌ Local chat (26B) failed: {e}")
        results["chat_gemma4_26b"] = False
    
    try:
        results["rag_pipeline_local"] = test_local_rag_pipeline()
    except Exception as e:
        print(f"❌ Local RAG pipeline failed: {e}")
        results["rag_pipeline_local"] = False
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (Local Ollama)")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'ALL PASSED ✅' if all_passed else 'SOME FAILED ❌'}")
    sys.exit(0 if all_passed else 1)
