from langchain_ollama import ChatOllama

def test_ollama():
    print("Testing Ollama gemma4:latest...")
    try:
        llm = ChatOllama(model="gemma4:latest")
        res = llm.invoke("Hello! Reply with one short sentence.")
        print(f"Success! Response: {res.content}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_ollama()
