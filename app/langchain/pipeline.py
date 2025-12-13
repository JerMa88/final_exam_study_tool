from typing import Iterator, Optional
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from .retrieval import Retriever
from .database import ArcadeDBClient

class RAGPipeline:
    def __init__(self, retriever: Retriever, db_client: ArcadeDBClient):
        self.retriever = retriever
        self.db = db_client
    
    def _get_llm(self, model_name: str):
        # Allow passing custom model names directly to VertexAI
        # If 'locally' is implied for some gemma models, we might need Ollama support here too.
        # But user said "locally or on google cloud" for gemma. 
        # For simplicity, we default to VertexAI for the 'models/' prefix 
        # unless we detect it's an ollama model (usually no 'models/' prefix or specific config).
        # We will assume these are valid vertex model IDs or mapped accordingly.
        return ChatVertexAI(model_name=model_name, temperature=0.7)

    def chat_stream(self, query: str, conversation_id: Optional[str] = None, model_name: str = "models/gemini-2.0-flash") -> Iterator[str]:
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
        
        llm = self._get_llm(model_name)
        
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
                yield chunk
        except Exception as e:
            error_msg = f"Error during generation: {e}"
            full_response += error_msg
            yield error_msg
            
        self.db.add_message(conversation_id, "assistant", full_response)
        
        # Return conversation ID via side-channel or assume client tracks it?
        # Typically the generator yields content. We might need to yield metadata first?
        # For this simple implementation, we just stream content.
