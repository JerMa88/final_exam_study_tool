import requests
import json
import base64
from typing import List, Dict, Any, Optional

class ArcadeDBClient:
    def __init__(self, host: str = "http://localhost", port: int = 2480, db_name: str = "study_tool"):
        self.base_url = f"{host}:{port}/api/v1"
        self.db_name = db_name
        self.auth = ("root", "securepassword")
        self.headers = {"Content-Type": "application/json"}
        self._ensure_db_exists()
        self._init_schema()

    def _ensure_db_exists(self):
        """Check if DB exists, create if not."""
        if not self._db_exists():
            print(f"Creating database {self.db_name}...")
            response = requests.post(
                f"{self.base_url}/server",
                json={"command": f"create database {self.db_name}"},
                auth=self.auth,
                headers=self.headers
            )
            if response.status_code != 200:
                print(f"Failed to create DB: {response.text}")
            else:
                print(f"Database {self.db_name} created.")

    def _db_exists(self) -> bool:
        """Check if database exists by attempting to connect."""
        try:
            url = f"{self.base_url}/query/{self.db_name}"
            payload = {"command": "SELECT 1", "language": "sql"}
            res = requests.post(url, json=payload, auth=self.auth, headers=self.headers)
            return res.status_code == 200
        except Exception as e:
            print(f"Error checking DB existence: {e}")
            return False

    def _init_schema(self):
        """Initialize Vertices and Edges."""
        commands = [
            # --- Document storage ---
            "CREATE VERTEX TYPE SourceFile IF NOT EXISTS",
            "CREATE PROPERTY SourceFile.filename STRING", 
            "CREATE PROPERTY SourceFile.type STRING", 
            "CREATE PROPERTY SourceFile.upload_date DATETIME",
            
            "CREATE VERTEX TYPE Chunk IF NOT EXISTS",
            "CREATE PROPERTY Chunk.content STRING",
            "CREATE PROPERTY Chunk.page_number INTEGER",
            "CREATE PROPERTY Chunk.embedding LIST",
            
            "CREATE INDEX ON Chunk (embedding) NOTUNIQUE",
            
            "CREATE EDGE TYPE HAS_CHUNK IF NOT EXISTS", 
            
            # --- Study Sessions ---
            "CREATE VERTEX TYPE StudySession IF NOT EXISTS",
            "CREATE PROPERTY StudySession.name STRING",
            "CREATE PROPERTY StudySession.created_at DATETIME",
            "CREATE PROPERTY StudySession.updated_at DATETIME",
            
            "CREATE EDGE TYPE STUDIES_FILE IF NOT EXISTS",      # StudySession -> SourceFile
            "CREATE EDGE TYPE HAS_CONVERSATION IF NOT EXISTS",  # StudySession -> Conversation
            
            # --- Conversations & Messages ---
            "CREATE VERTEX TYPE Conversation IF NOT EXISTS",
            "CREATE PROPERTY Conversation.started_at DATETIME",
            
            "CREATE VERTEX TYPE Message IF NOT EXISTS",
            "CREATE PROPERTY Message.role STRING",
            "CREATE PROPERTY Message.content STRING",
            "CREATE PROPERTY Message.timestamp DATETIME",
            
            "CREATE EDGE TYPE HAS_MESSAGE IF NOT EXISTS"
        ]
        
        for cmd in commands:
            self.execute_command(cmd, language="sql")

    def execute_command(self, command: str, language: str = "sql") -> Dict:
        url = f"{self.base_url}/command/{self.db_name}"
        payload = {"command": command, "language": language}
        response = requests.post(url, json=payload, auth=self.auth, headers=self.headers)
        
        if response.status_code != 200:
            error_text = response.text
            # Benign errors to suppress during schema init
            if "already exists" in error_text:
                return {}
            
            print(f"Command failed ({language}): {command} -> {error_text}")
            return {}
        return response.json()

    def query(self, query: str, language: str = "sql", params: Dict = None) -> List[Dict]:
        url = f"{self.base_url}/query/{self.db_name}"
        payload = {"command": query, "language": language}
        if params:
            payload["params"] = params
            
        response = requests.post(url, json=payload, auth=self.auth, headers=self.headers)
        if response.status_code != 200:
             print(f"Query failed ({url}): {query} -> {response.text}")
             return []
        return response.json().get("result", [])

    # ─── Document Methods ────────────────────────────────────────────

    def document_exists(self, filename: str) -> bool:
        """Check if a document with the given filename already exists."""
        query = "SELECT FROM SourceFile WHERE filename = ?"
        res = self.query(query, params={"0": filename})
        return len(res) > 0

    def get_document_rid(self, filename: str) -> str:
        """Get the RID of a SourceFile by filename."""
        res = self.query("SELECT @rid FROM SourceFile WHERE filename = ?", params={"0": filename})
        if res:
            return res[0].get("@rid", "")
        return ""

    def insert_document(self, filename: str, doc_type: str) -> str:
        """Insert document metadata and return RID."""
        cmd = "INSERT INTO SourceFile SET filename = ?, type = ?, upload_date = sysdate()"
        payload = {
            "command": cmd,
            "language": "sql",
            "params": {
                "0": filename,
                "1": doc_type
            }
        }
        url = f"{self.base_url}/command/{self.db_name}"
        response = requests.post(url, json=payload, auth=self.auth, headers=self.headers)
        if response.status_code == 200:
            res = response.json().get("result", [])
            if res:
                return res[0].get("@rid")
        else:
            print(f"Insert document failed: {response.text}")
        return ""

    def insert_chunk(self, doc_rid: str, content: str, page_num: int, embedding: List[float]):
        """Insert chunk and link to document."""
        cmd_chunk = "INSERT INTO Chunk SET content = ?, page_number = ?, embedding = ?"
        payload_chunk = {
            "command": cmd_chunk,
            "language": "sql",
            "params": {
                "0": content,
                "1": page_num,
                "2": embedding
            }
        }
        url = f"{self.base_url}/command/{self.db_name}"
        res = requests.post(url, json=payload_chunk, auth=self.auth, headers=self.headers)
        chunk_rid = None
        if res.status_code == 200:
            r = res.json().get("result", [])
            if r:
                chunk_rid = r[0].get("@rid")
        else:
            print(f"Insert chunk failed: {res.text}")
        
        if chunk_rid and doc_rid:
            cmd_link = f"CREATE EDGE HAS_CHUNK FROM {doc_rid} TO {chunk_rid}"
            self.execute_command(cmd_link)

    def get_all_documents(self) -> List[Dict]:
        """Get all ingested documents."""
        res = self.query("SELECT @rid, filename, type, upload_date FROM SourceFile ORDER BY upload_date DESC")
        return res

    # ─── Study Session Methods ───────────────────────────────────────

    def create_session(self, name: str) -> Dict:
        """Create a new study session and return its details."""
        cmd = "INSERT INTO StudySession SET name = ?, created_at = sysdate(), updated_at = sysdate()"
        payload = {
            "command": cmd,
            "language": "sql",
            "params": {"0": name}
        }
        url = f"{self.base_url}/command/{self.db_name}"
        response = requests.post(url, json=payload, auth=self.auth, headers=self.headers)
        if response.status_code == 200:
            res = response.json().get("result", [])
            if res:
                return {"rid": res[0].get("@rid"), "name": name}
        return {}

    def list_sessions(self) -> List[Dict]:
        """List all study sessions, newest first."""
        res = self.query(
            "SELECT @rid, name, created_at, updated_at FROM StudySession ORDER BY updated_at DESC"
        )
        return res

    def get_session(self, session_rid: str) -> Dict:
        """Get a single session by RID."""
        res = self.query(f"SELECT @rid, name, created_at, updated_at FROM {session_rid}")
        if res:
            return res[0]
        return {}

    def delete_session(self, session_rid: str):
        """Delete a session and its edges (conversations/messages remain for safety)."""
        # Delete edges first
        self.execute_command(f"DELETE EDGE STUDIES_FILE WHERE out = {session_rid}")
        self.execute_command(f"DELETE EDGE HAS_CONVERSATION WHERE out = {session_rid}")
        # Delete session vertex
        self.execute_command(f"DELETE FROM {session_rid}")

    def rename_session(self, session_rid: str, new_name: str):
        """Rename a study session."""
        self.execute_command(
            f"UPDATE {session_rid} SET name = ?, updated_at = sysdate()",
            # Using direct string embed since UPDATE with params can be tricky
        )
        # Fallback: use direct SQL with safe name
        cmd = f"UPDATE {session_rid} SET name = '{new_name.replace(chr(39), chr(39)+chr(39))}', updated_at = sysdate()"
        self.execute_command(cmd)

    def link_file_to_session(self, session_rid: str, file_rid: str):
        """Create STUDIES_FILE edge from session to source file."""
        # Check if link already exists
        existing = self.query(
            f"SELECT FROM STUDIES_FILE WHERE out = {session_rid} AND in = {file_rid}"
        )
        if not existing:
            self.execute_command(f"CREATE EDGE STUDIES_FILE FROM {session_rid} TO {file_rid}")
            # Update session timestamp
            self.execute_command(f"UPDATE {session_rid} SET updated_at = sysdate()")

    def unlink_file_from_session(self, session_rid: str, file_rid: str):
        """Remove STUDIES_FILE edge."""
        self.execute_command(
            f"DELETE EDGE STUDIES_FILE WHERE out = {session_rid} AND in = {file_rid}"
        )

    def get_session_files(self, session_rid: str) -> List[Dict]:
        """Get all source files linked to a session."""
        res = self.query(
            f"SELECT expand(in) FROM STUDIES_FILE WHERE out = {session_rid}"
        )
        # The expand might return SourceFile records directly
        if not res:
            # Fallback: traverse
            res = self.query(
                f"SELECT @rid, filename, type FROM (TRAVERSE out('STUDIES_FILE') FROM {session_rid}) WHERE @type = 'SourceFile'"
            )
        return res

    def get_session_file_rids(self, session_rid: str) -> List[str]:
        """Get RIDs of source files linked to a session."""
        files = self.get_session_files(session_rid)
        return [f.get("@rid", "") for f in files if f.get("@rid")]

    # ─── Conversation Methods (Session-Scoped) ───────────────────────

    def create_conversation(self, session_rid: str = None) -> str:
        """Create a new conversation, optionally linked to a session."""
        cmd = "INSERT INTO Conversation SET started_at = sysdate()"
        res = self.execute_command(cmd)
        conv_rid = ""
        if res and "result" in res:
            conv_rid = res["result"][0].get("@rid", "")
        
        # Link to session if provided
        if conv_rid and session_rid:
            self.execute_command(f"CREATE EDGE HAS_CONVERSATION FROM {session_rid} TO {conv_rid}")
            self.execute_command(f"UPDATE {session_rid} SET updated_at = sysdate()")
        
        return conv_rid

    def get_session_conversation(self, session_rid: str) -> str:
        """Get the conversation RID for a session (most recent one)."""
        res = self.query(
            f"SELECT @rid FROM (TRAVERSE out('HAS_CONVERSATION') FROM {session_rid}) "
            f"WHERE @type = 'Conversation' ORDER BY started_at DESC LIMIT 1"
        )
        if res:
            return res[0].get("@rid", "")
        return ""

    def add_message(self, conv_rid: str, role: str, content: str):
        """Add a message to a conversation."""
        cmd = "INSERT INTO Message SET role = ?, content = ?, timestamp = sysdate()"
        payload = {"command": cmd, "language": "sql", "params": {"0": role, "1": content}}
        url = f"{self.base_url}/command/{self.db_name}"
        res = requests.post(url, json=payload, auth=self.auth, headers=self.headers)
        msg_rid = None
        if res.status_code == 200:
            result = res.json().get("result", [])
            if result:
                msg_rid = result[0].get("@rid")
        
        if msg_rid and conv_rid:
            self.execute_command(f"CREATE EDGE HAS_MESSAGE FROM {conv_rid} TO {msg_rid}")

    def get_conversation_messages(self, conv_rid: str) -> List[Dict]:
        """Get all messages in a conversation, ordered by timestamp."""
        res = self.query(
            f"SELECT role, content, timestamp FROM ("
            f"TRAVERSE out('HAS_MESSAGE') FROM {conv_rid}"
            f") WHERE @type = 'Message' ORDER BY timestamp ASC"
        )
        return res

    def get_session_messages(self, session_rid: str) -> List[Dict]:
        """Get all messages for a session (via its conversation)."""
        conv_rid = self.get_session_conversation(session_rid)
        if not conv_rid:
            return []
        return self.get_conversation_messages(conv_rid)

    # ─── Similarity Search (Session-Scoped) ──────────────────────────

    def similarity_search(self, embedding: List[float], limit: int = 5, 
                          file_rids: List[str] = None) -> List[Dict]:
        """
        Search similar chunks using client-side Cosine Similarity.
        If file_rids is provided, only search chunks belonging to those files.
        """
        if file_rids:
            # Build query to only fetch chunks from specified files
            rid_list = ", ".join(file_rids)
            query = (
                f"SELECT @rid, content, page_number, embedding FROM Chunk "
                f"WHERE @rid IN (SELECT expand(out('HAS_CHUNK')) FROM [{rid_list}])"
            )
            # Simpler approach: traverse from files
            query = (
                f"SELECT @rid, content, page_number, embedding FROM ("
                f"TRAVERSE out('HAS_CHUNK') FROM [{rid_list}]"
                f") WHERE @type = 'Chunk'"
            )
        else:
            query = "SELECT @rid, content, page_number, embedding FROM Chunk"
        
        res = self.query(query)
        
        if not res:
            return []

        import numpy as np
        
        query_vec = np.array(embedding)
        norm_q = np.linalg.norm(query_vec)
        if norm_q == 0:
            return []
            
        scored_chunks = []
        for r in res:
            vec = r.get("embedding")
            if not vec:
                continue
            if isinstance(vec, str):
                continue
                
            doc_vec = np.array(vec)
            norm_d = np.linalg.norm(doc_vec)
            
            if norm_d == 0:
                score = 0
            else:
                score = np.dot(query_vec, doc_vec) / (norm_q * norm_d)
            
            scored_chunks.append({
                "content": r.get("content"), 
                "page_number": r.get("page_number"), 
                "score": score
            })
            
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:limit]
