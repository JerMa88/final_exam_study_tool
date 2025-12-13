import os
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI

def test_model(model_name):
    print(f"Testing {model_name}...")
    try:
        llm = ChatGoogleGenerativeAI(model=model_name)
        res = llm.invoke("Hello!")
        print(f"Success! Response: {res.content}")
        return True
    except Exception as e:
        print(f"Failed: {e}")
        return False

if __name__ == "__main__":
    test_model("models/gemini-2.0-flash")
