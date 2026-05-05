from typing import Iterator, Optional
import os
from langchain_google_vertexai import ChatVertexAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from .retrieval import Retriever
from .database import ArcadeDBClient

class RAGPipeline:
    def __init__(self, retriever: Retriever, db_client: ArcadeDBClient):
        self.retriever = retriever
        self.db = db_client
    
    def _get_llm(self, model_name: str, llm_provider: str = "vertex"):
        if llm_provider == "ollama":
            # For Ollama, the model name is passed directly (e.g. "gemma:2b")
            # But frontend sends "models/gemma-3n-e4b-it", we need to map or strip
            # Assuming simplified mapping or user passes exact Ollama tag if they know it.
            # Let's strip "models/" prefix if present for Ollama usage
            clean_name = model_name.replace("models/", "")
            
            from .utils import ensure_ollama_model
            ensure_ollama_model(clean_name)
            
            return ChatOllama(model=clean_name)
        else:
            # Check for API Key to determine which Google client to use
            if os.getenv("GOOGLE_API_KEY"):
                # Use GenAI (AI Studio) client
                return ChatGoogleGenerativeAI(model=model_name)
            else:
                # Default to Vertex AI (requires gcloud auth)
                return ChatVertexAI(model_name=model_name)

    def chat_stream(self, query: str, conversation_id: Optional[str] = None, model_name: str = "models/gemini-2.0-flash", llm_provider: str = "vertex") -> Iterator[str]:
        import json
        
        # 1. Retrieve
        context_docs = self.retriever.search(query, limit=5)
        context_str = "\n\n".join(context_docs)
        
        # 2. Persist User Message
        if not conversation_id:
             conversation_id = self.db.create_conversation()
        
        self.db.add_message(conversation_id, "user", query)

        # 3. Setup Chain
        template = """Answer the question regarding the study materials.
        
        Context:
        {context}
        
        Question: {question}
        
        Instructions: 
        1. Use the provided Context to answer the Question.
        2. If the answer is not found in the Context, you may answer from your own general knowledge, but explicitly state: "Note: This answer is based on general knowledge as it was not found in the provided documents."
        
        Response:"""
        prompt = ChatPromptTemplate.from_template(template)
        
        # Debug: Yield prompt info
        filled_prompt = prompt.format(context=context_str, question=query)
        yield json.dumps({"type": "debug", "prompt": filled_prompt}) + "\n"
        
        llm = self._get_llm(model_name, llm_provider)
        
        chain = (
            {"context": lambda x: context_str, "question": lambda x: x}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # 4. Stream and Persist AI Message
        full_response = ""
        try:
            for chunk in chain.stream(query):
                full_response += chunk
                yield json.dumps({"type": "token", "content": chunk}) + "\n"
        except Exception as e:
            error_msg = f"Error during generation: {e}"
            full_response += error_msg
            yield json.dumps({"type": "token", "content": error_msg}) + "\n"
            
        self.db.add_message(conversation_id, "assistant", full_response)
