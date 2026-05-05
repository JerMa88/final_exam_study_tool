"""
Cheat Sheet Generation Pipeline.

Loops over user-provided exam topics, retrieves context via RAG for each,
summarizes per-topic, then assembles into a dense cheat sheet.
Also supports iterative refinement through user instructions.
"""
from typing import Iterator, Optional, List
import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .retrieval import Retriever
from .database import ArcadeDBClient

# ─── Prompt Templates ────────────────────────────────────────────────

TOPIC_SUMMARIZE_PROMPT = """You are creating a cheat sheet for a final exam.
Summarize the following topic using the provided context from course materials.

Topic: {topic}

Context from course materials:
{context}

Instructions:
1. Extract ALL key equations, formulas, and mathematical notation. Use LaTeX ($...$) for math.
2. Include definitions, relationships, and important properties.
3. Cite sources using [cite: filename, page_number] format based on the source info in the context.
4. Be maximally dense — this must fit on a cheat sheet.
5. Use bullet points, not paragraphs.
6. DO NOT include introductory or filler text.
7. If the context contains relevant equations or formulas, reproduce them EXACTLY in LaTeX.

Summary:"""

ASSEMBLE_PROMPT = """You are assembling a final exam cheat sheet from per-topic summaries.

Per-topic summaries:
{summaries}

Instructions:
1. Organize into numbered sections using ## **N. Topic Name** format.
2. Merge overlapping content between topics — do not repeat the same equation twice.
3. Preserve ALL equations, citations, and technical details from the summaries.
4. Use consistent formatting: section headers → bullet points → indented sub-bullets for details.
5. Maximize information density. Every line should contain testable knowledge.
6. Use LaTeX ($...$) for all mathematical notation.
7. Keep citation format as [cite: filename, page_number].
8. Do NOT add any introductory paragraph, preamble, or closing summary.

Complete Cheat Sheet:"""

REFINE_PROMPT = """You are refining a final exam cheat sheet based on user feedback.

Current Cheat Sheet:
{current_cheatsheet}

{context_section}

User Request: {instruction}

Instructions:
1. Apply the user's requested changes to the cheat sheet.
2. Preserve all existing formatting, equations, and citations unless the user asks to remove them.
3. If adding new content, maintain the same dense, citation-rich, bullet-point style with LaTeX math.
4. Return the COMPLETE updated cheat sheet (not just the changes).
5. Keep citation format as [cite: filename, page_number].

Updated Cheat Sheet:"""


class CheatSheetPipeline:
    """Pipeline for generating and refining exam cheat sheets using RAG."""

    def __init__(self, retriever: Retriever, db_client: ArcadeDBClient):
        self.retriever = retriever
        self.db = db_client

    def _get_llm(self, model_name: str, llm_provider: str = "vertex"):
        """Get an LLM instance (mirrors RAGPipeline._get_llm)."""
        if llm_provider == "ollama":
            clean_name = model_name.replace("models/", "")
            from .utils import ensure_ollama_model
            ensure_ollama_model(clean_name)
            return ChatOllama(model=clean_name)
        else:
            if os.getenv("GOOGLE_API_KEY"):
                return ChatGoogleGenerativeAI(model=model_name)
            else:
                return ChatVertexAI(model_name=model_name)

    def _summarize_topic(
        self, 
        topic: str, 
        llm, 
        file_rids: Optional[List[str]] = None,
    ) -> str:
        """Retrieve context for a topic and generate a dense summary."""
        # Retrieve top-8 chunks for this topic
        context_docs = self.retriever.search(topic, limit=8, file_rids=file_rids)
        context_str = "\n\n".join(context_docs)

        if not context_str.strip():
            return f"* _No relevant material found for topic: {topic}_"

        prompt = ChatPromptTemplate.from_template(TOPIC_SUMMARIZE_PROMPT)
        chain = (
            {"topic": lambda x: topic, "context": lambda x: context_str}
            | prompt
            | llm
            | StrOutputParser()
        )

        try:
            result = chain.invoke(topic)
            return result.strip()
        except Exception as e:
            return f"* _Error summarizing topic '{topic}': {e}_"

    def generate_stream(
        self,
        topics: List[str],
        session_rid: str,
        model_name: str = "models/gemini-2.0-flash",
        llm_provider: str = "vertex",
        file_rids: Optional[List[str]] = None,
    ) -> Iterator[str]:
        """
        Generate a cheat sheet by looping over topics.
        
        Yields JSON-line events:
          {"type": "topic_progress", "topic": ..., "index": ..., "total": ...}
          {"type": "topic_result", "topic": ..., "summary": ...}
          {"type": "cheatsheet", "content": ..., "cheatsheet_rid": ...}
          {"type": "error", "message": ...}
        """
        total = len(topics)
        llm = self._get_llm(model_name, llm_provider)

        # Phase 1: Per-topic summarization
        topic_summaries = []
        for i, topic in enumerate(topics):
            yield json.dumps({
                "type": "topic_progress",
                "topic": topic,
                "index": i + 1,
                "total": total,
                "status": "processing"
            }) + "\n"

            summary = self._summarize_topic(topic, llm, file_rids)
            topic_summaries.append({"topic": topic, "summary": summary})

            yield json.dumps({
                "type": "topic_result",
                "topic": topic,
                "index": i + 1,
                "total": total,
                "summary": summary,
                "status": "done"
            }) + "\n"

        # Phase 2: Assembly
        yield json.dumps({
            "type": "topic_progress",
            "topic": "Assembling final cheat sheet...",
            "index": total,
            "total": total,
            "status": "assembling"
        }) + "\n"

        summaries_text = ""
        for ts in topic_summaries:
            summaries_text += f"### {ts['topic']}\n{ts['summary']}\n\n"

        prompt = ChatPromptTemplate.from_template(ASSEMBLE_PROMPT)
        chain = (
            {"summaries": lambda x: summaries_text}
            | prompt
            | llm
            | StrOutputParser()
        )

        try:
            final_cheatsheet = chain.invoke("assemble")
        except Exception as e:
            yield json.dumps({
                "type": "error",
                "message": f"Assembly failed: {e}"
            }) + "\n"
            return

        # Phase 3: Persist to DB
        topics_json = json.dumps(topics)
        
        # Check if session already has a cheat sheet — update it instead of creating new
        existing = self.db.get_session_cheatsheet(session_rid)
        if existing and existing.get("@rid"):
            cs_rid = existing["@rid"]
            self.db.update_cheatsheet(cs_rid, final_cheatsheet)
        else:
            cs_rid = self.db.create_cheatsheet(session_rid, topics_json, final_cheatsheet)

        yield json.dumps({
            "type": "cheatsheet",
            "content": final_cheatsheet,
            "cheatsheet_rid": cs_rid,
        }) + "\n"

    def refine_stream(
        self,
        instruction: str,
        session_rid: str,
        model_name: str = "models/gemini-2.0-flash",
        llm_provider: str = "vertex",
        file_rids: Optional[List[str]] = None,
    ) -> Iterator[str]:
        """
        Refine an existing cheat sheet based on user instruction.
        
        Yields JSON-line events:
          {"type": "refine_progress", "status": ...}
          {"type": "cheatsheet", "content": ..., "cheatsheet_rid": ...}
          {"type": "error", "message": ...}
        """
        # Load current cheat sheet
        existing = self.db.get_session_cheatsheet(session_rid)
        if not existing or not existing.get("content"):
            yield json.dumps({
                "type": "error",
                "message": "No cheat sheet found for this session. Generate one first."
            }) + "\n"
            return

        current_content = existing["content"]
        cs_rid = existing["@rid"]

        yield json.dumps({
            "type": "refine_progress",
            "status": "Retrieving additional context..."
        }) + "\n"

        # Optionally retrieve additional context if the instruction implies new info needed
        context_section = ""
        context_docs = self.retriever.search(instruction, limit=8, file_rids=file_rids)
        if context_docs:
            context_str = "\n\n".join(context_docs)
            context_section = f"Additional context from course materials (use if relevant to the request):\n{context_str}"

        yield json.dumps({
            "type": "refine_progress",
            "status": "Applying refinements..."
        }) + "\n"

        llm = self._get_llm(model_name, llm_provider)
        prompt = ChatPromptTemplate.from_template(REFINE_PROMPT)
        chain = (
            {
                "current_cheatsheet": lambda x: current_content,
                "context_section": lambda x: context_section,
                "instruction": lambda x: instruction,
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        try:
            refined = chain.invoke(instruction)
        except Exception as e:
            yield json.dumps({
                "type": "error",
                "message": f"Refinement failed: {e}"
            }) + "\n"
            return

        # Update DB
        self.db.update_cheatsheet(cs_rid, refined)

        yield json.dumps({
            "type": "cheatsheet",
            "content": refined,
            "cheatsheet_rid": cs_rid,
        }) + "\n"
