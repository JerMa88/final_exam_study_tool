from typing import Iterator, Optional
from langchain_google_vertexai import ChatVertexAI
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from .retrieval import Retriever
from .database import ArcadeDBClient

class RAGPipeline:
    def __init__(self, retriever: Retriever, db_client: ArcadeDBClient):
        self.retriever = retriever
        self.db = db_client
    
    def _get_llm(self, model_name: str, provider: str = "vertex"):
        if provider == "ollama":
            # Strip 'models/' prefix if present for local Ollama usage
            local_model = model_name.replace("models/", "")
            return ChatOllama(model=local_model, temperature=0.7)
        # Default to VertexAI
        return ChatVertexAI(model_name=model_name, temperature=0.7)

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
        template = """Answer the question based only on the following context:
        {context}
        
        Question: {question}
        
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
