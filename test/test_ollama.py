from langchain_community.chat_models import ChatOllama

def test_ollama():
    print("Testing Ollama gemma:2b...")
    try:
        llm = ChatOllama(model="gemma:2b")
        res = llm.invoke("Hello!")
        print(f"Success! Response: {res.content}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_ollama()
