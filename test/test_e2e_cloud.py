"""
End-to-End Test: Cloud Gemini API
Tests embedding via GoogleGenerativeAIEmbeddings and chat via ChatGoogleGenerativeAI.
Requires: GOOGLE_API_KEY set in .env
"""
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

def test_api_key_present():
    """Check that API key is configured."""
    api_key = os.getenv("GOOGLE_API_KEY")
    assert api_key and len(api_key) > 10, "GOOGLE_API_KEY not set or too short in .env"
    print(f"✅ API Key found (length={len(api_key)})")
    return True

def test_cloud_embedding():
    """Test Google GenAI embedding model."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    
    print("\n--- Testing Cloud Embedding (gemini-embedding-001) ---")
    start = time.time()
    embedder = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vec = embedder.embed_query("What is quantum entanglement?")
    elapsed = time.time() - start
    
    assert isinstance(vec, list), f"Expected list, got {type(vec)}"
    assert len(vec) > 0, "Empty embedding vector"
    print(f"✅ Embedding success: dim={len(vec)}, time={elapsed:.2f}s")
    print(f"   First 5 values: {vec[:5]}")
    return vec

def test_cloud_chat():
    """Test Google GenAI chat model (tries gemini-2.5-flash, falls back to gemini-2.0-flash)."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    models_to_try = ["models/gemini-2.5-flash", "models/gemini-2.0-flash"]
    
    for model_name in models_to_try:
        print(f"\n--- Testing Cloud Chat ({model_name}) ---")
        start = time.time()
        try:
            llm = ChatGoogleGenerativeAI(model=model_name)
            res = llm.invoke("What is 2+2? Reply with just the number.")
            elapsed = time.time() - start
            
            assert res.content, "Empty response"
            print(f"✅ Chat success: model={model_name}, response='{res.content.strip()}', time={elapsed:.2f}s")
            return res.content
        except Exception as e:
            print(f"⚠️  {model_name} failed: {e}")
            continue
    
    raise RuntimeError("All cloud chat models failed")

def test_cloud_rag_pipeline():
    """Test the full RAG pipeline with cloud provider (no DB required)."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    print("\n--- Testing Cloud RAG Pipeline (embed + retrieve mock + generate) ---")
    
    # 1. Embed a query
    embedder = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    query = "Explain Grover's algorithm"
    query_vec = embedder.embed_query(query)
    
    # 2. Mock context (simulating retrieval)
    mock_context = """
    [Page 5] Grover's algorithm is a quantum algorithm for searching an unsorted database 
    with N entries in O(√N) time, providing a quadratic speedup over classical algorithms.
    It uses amplitude amplification to increase the probability of measuring the correct answer.
    """
    
    # 3. Generate response (try multiple models)
    template = """Answer the question using the provided context.
    Context: {context}
    Question: {question}
    Response:"""
    prompt = ChatPromptTemplate.from_template(template)
    
    for model_name in ["models/gemini-2.5-flash", "models/gemini-2.0-flash"]:
        try:
            llm = ChatGoogleGenerativeAI(model=model_name)
            
            chain = (
                {"context": lambda x: mock_context, "question": lambda x: x}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            start = time.time()
            response = chain.invoke(query)
            elapsed = time.time() - start
            
            assert len(response) > 20, f"Response too short: {response}"
            print(f"✅ RAG pipeline success: model={model_name}, time={elapsed:.2f}s")
            print(f"   Response preview: {response[:200]}...")
            return True
        except Exception as e:
            print(f"⚠️  RAG with {model_name} failed: {e}")
            continue
    
    raise RuntimeError("All cloud models failed for RAG pipeline")

if __name__ == "__main__":
    print("=" * 60)
    print("E2E TEST: Cloud Gemini API")
    print("=" * 60)
    
    results = {}
    
    try:
        results["api_key"] = test_api_key_present()
    except AssertionError as e:
        print(f"❌ API Key check failed: {e}")
        print("Aborting cloud tests — no API key.")
        sys.exit(1)
    
    try:
        results["embedding"] = test_cloud_embedding() is not None
    except Exception as e:
        print(f"❌ Cloud embedding failed: {e}")
        results["embedding"] = False
    
    try:
        results["chat"] = test_cloud_chat() is not None
    except Exception as e:
        print(f"❌ Cloud chat failed: {e}")
        results["chat"] = False
    
    try:
        results["rag_pipeline"] = test_cloud_rag_pipeline()
    except Exception as e:
        print(f"❌ Cloud RAG pipeline failed: {e}")
        results["rag_pipeline"] = False
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (Cloud Gemini)")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'ALL PASSED ✅' if all_passed else 'SOME FAILED ❌'}")
    sys.exit(0 if all_passed else 1)
