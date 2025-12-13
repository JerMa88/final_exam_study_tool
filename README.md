# Final Exam Study Tool

A locally-hosted Retrieval Augmented Generation (RAG) agent designed to help you study for final exams. It ingests course materials (Slides, Textbooks, Papers, Notes), stores them in a graph-vector database (ArcadeDB), and allows you to chat with or search through your materials using advanced LLMs (Gemini, Gemma).

## Key Features
- **Multi-Modal Ingestion**: Automatically processes PDFs and Text files, categorizing them as Slides, Textbooks, or Papers.
- **Hybrid Storage**: Uses **ArcadeDB** to store document structure and vector embeddings (graph-based retrieval ready).
- **Flexible LLM Support**:
  - **Google Gemini** (via API Key): Access `gemini-2.0-flash`, `gemini-2.5-pro` without complex Cloud setup.
  - **Ollama** (Local): Run `gemma:2b` and other open models locally for privacy and offline usage.
- **Smart Search**: Client-side cosine similarity search to find relevant course concepts instantly.
- **Interactive UI**: Built with **Streamlit** for a seamless Chat and Search experience.

---

## Prerequisites
- **Python 3.9+**
- **Docker** & **Docker Compose** (for ArcadeDB)
- **Ollama** (optional, for local models)
- **Google API Key** (recommended for best chat performance)

---

## Setup Guide

### 1. Environment Configuration

Clone the repository and create a `.env` file in the root directory:

```bash
touch .env
```

Add your Google API Key to `.env`:
```text
GOOGLE_API_KEY=your_actual_api_key_here
```
> **Note**: Get a key from [Google AI Studio](https://aistudio.google.com/). If you prefer local-only, you can skip this but must use Ollama.

### 2. Start the Database (ArcadeDB)

We use ArcadeDB running in Docker. Start it with:

```bash
docker compose up -d
```
*Port 2480 (HTTP) and 2424 (Binary) will be exposed.*

### 3. Setup Ollama (Local Models) - Recommended

If you want to use local embeddings (Gemma) or chat models:
1. [Install Ollama](https://ollama.com/).
2. Pull the required models:
   ```bash
   ollama pull gemma:2b
   ```
   *Note: The system uses `gemma:2b` for local embeddings and chat.*
3. Start the server:
   ```bash
   ollama serve
   ```

### 4. Install Dependencies

Create a virtual environment and install the Python packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 5. Running the Application

You need to run both the Backend (API) and the Frontend (UI).

**Terminal 1: Backend (FastAPI)**
```bash
source .venv/bin/activate
uvicorn app.langchain.main:app --host 0.0.0.0 --port 8000 --reload
```
*On startup, the backend checks `database/PDFs` and automatically ingests any new files.*

**Terminal 2: Frontend (Streamlit)**
```bash
source .venv/bin/activate
streamlit run app/streamlit/app.py
```
*Access the UI at [http://localhost:8501](http://localhost:8501).*

---

## Usage Documentation

### Ingestion
- **Automatic**: Place your PDF/TXT files in `database/PDFs/slides`, `textbooks`, or `papers`. Restart the backend to ingest them.
- **Manual**: Use the **"Upload Document"** sidebar in the Streamlit UI to add files dynamically.

### Chat Interface
1. **Select Provider**: Choose "Vertex" (Google API) or "Ollama" in the sidebar.
2. **Select Model**:
   - **Vertex**: `gemini-2.0-flash` (Fast), `gemini-2.5-pro` (Reasoning), `gemma-3-12b-it`.
   - **Ollama**: `gemma:2b` (Local).
3. **Ask Questions**: Type queries like "Explain Grover's Algorithm from the slides". The agent uses RAG to fetch context and cite sources (e.g., `(Page 3)`).

### Search Interface
1. Switch to the **Search** tab.
2. Enter a concept (e.g., "Tensor Products").
3. View exact text matches and their source pages.
   - *Note*: Ensure your selected Provider matches the one used for ingestion for best results (Ollama `gemma:2b` is the current default for ingestion).

---

## Troubleshooting

- **"Model not found" error**:
  Ensure you have pulled the model in Ollama: `ollama pull gemma:2b`.
- **Search returns no results**:
  If you switched embedding models (e.g., from Nomic to Gemma), you must re-ingest your data. Stop the app, clear `database/ArcadeDB/arcadedb-*/databases`, and restart the backend.
- **500 Error on Chat**:
  Check the backend logs terminal. Ensure your `GOOGLE_API_KEY` is valid if using Vertex/Gemini models.

---

## Architecture
- **Backend**: FastAPI
- **LLM Orchestration**: LangChain
- **Database**: ArcadeDB (Vertex-Type Schema)
- **Frontend**: Streamlit