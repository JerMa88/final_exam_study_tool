# final_exam_study_tool
Help you study for your final

## Setup Documentation

### ArcadeDB (Docker Setup)

We use Docker to run ArcadeDB. This requires Docker and Docker Compose to be installed.

1. **Configuration**: Create a `docker-compose.yml` file in the root directory (already provided in the repo):

   ```yaml
   version: '3.8'

   services:
     arcadedb:
       image: ghcr.io/arcadedata/arcadedb:latest
       container_name: arcadedb
       environment:
         - ARCADEDB_ROOT_PASSWORD=securepassword
         - JAVA_OPTS=-Darcadedb.server.rootPassword=securepassword
       ports:
         - "2480:2480" # HTTP
         - "2424:2424" # Binary
       volumes:
         - ./database/ArcadeDB/arcadedb-24.10.1/databases:/home/arcadedb/databases
         - ./database/ArcadeDB/arcadedb-24.10.1/config:/home/arcadedb/config
         - ./database/ArcadeDB/arcadedb-24.10.1/log:/home/arcadedb/log
       restart: unless-stopped
   ```

2. **Start the Server**:
   ```bash
   docker compose up -d
   ```

3. **Verify**:
   The server will be available at [http://localhost:2480](http://localhost:2480).
   
   Login credentials:
   - **User**: `root`
   - **Password**: `securepassword`


### Backend API (FastAPI + LangChain)

The backend provides PDF ingestion, embedding (Vertex/Ollama), and a RAG chat interface.

1. **Environment Setup**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Running the App**:
   ```bash
   uvicorn app.langchain.main:app --reload
   ```

3. **API Usage**:
   - **Upload PDF**:
     ```bash
     curl -X POST -F "file=@/path/to/slides.pdf" -F "type=slide" http://localhost:8000/upload
     ```
   - **Chat**:
     ```bash
     curl -X POST -H "Content-Type: application/json" -d '{"message": "Explain this slide", "model": "models/gemini-2.0-flash"}' http://localhost:8000/chat
     ```

   **Supported Models**:
   - `models/gemma-3n-e4b-it`
   - `models/gemma-3-12b-it`
   - `models/gemma-3-27b-it`
   - `models/gemini-2.0-flash`
   - `models/gemini-3-pro-preview`