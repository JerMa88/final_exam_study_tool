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
        # Define schema commands
        # Note: IF NOT EXISTS might not be supported for all commands in this version's SQL parser
        # We will attempt creation and ignore errors for now (simulating if not exists)
        commands = [
            "CREATE VERTEX TYPE SourceFile IF NOT EXISTS",
            "CREATE PROPERTY SourceFile.filename STRING", 
            "CREATE PROPERTY SourceFile.type STRING", 
            "CREATE PROPERTY SourceFile.upload_date DATETIME",
            
            "CREATE VERTEX TYPE Chunk IF NOT EXISTS",
            "CREATE PROPERTY Chunk.content STRING",
            "CREATE PROPERTY Chunk.page_number INTEGER",
            "CREATE PROPERTY Chunk.embedding LIST", # FLOAT_LIST might be LIST in simplified SQL
            
            # Index syntax: CREATE INDEX ON <type> (<prop>) ... (Nameless seems safer for some parsers)
            "CREATE INDEX ON Chunk (embedding) NOTUNIQUE",
            
            "CREATE EDGE TYPE HAS_CHUNK IF NOT EXISTS", 
            
            "CREATE VERTEX TYPE Conversation IF NOT EXISTS",
            "CREATE PROPERTY Conversation.started_at DATETIME",
            
            "CREATE VERTEX TYPE Message IF NOT EXISTS",
            "CREATE PROPERTY Message.role STRING",
            "CREATE PROPERTY Message.content STRING",
            "CREATE PROPERTY Message.timestamp DATETIME",
            
            "CREATE EDGE TYPE HAS_MESSAGE IF NOT EXISTS"
        ]
        
        for cmd in commands:
            # sqlscript can be picky for single DDLs, use sql
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

    def document_exists(self, filename: str) -> bool:
        """Check if a document with the given filename already exists."""
        query = "SELECT FROM SourceFile WHERE filename = ?"
        # params keys for SQL usually parameter placeholders like ? or :param.
        # ArcadeDB HTTP API with 'sql' language and 'params' map usually uses ? or positional 0,1..
        # Let's stick to the map using "0" as key for first ?
        res = self.query(query, params={"0": filename})
        return len(res) > 0

    def insert_document(self, filename: str, doc_type: str) -> str:
        """Insert document metadata and return RID."""
        cmd = "INSERT INTO SourceFile SET filename = ?, type = ?, upload_date = sysdate()"
        # ArcadeDB SQL params usually passed in parameters or embedded. 
        # For simplicity, we'll embed safely or use params if API supports it standardly.
        # The HTTP API supports 'params' map.
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
        # 1. Insert Chunk
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
        
        # 2. Link
        if chunk_rid and doc_rid:
            cmd_link = f"CREATE EDGE HAS_CHUNK FROM {doc_rid} TO {chunk_rid}"
            self.execute_command(cmd_link)
            
    def similarity_search(self, embedding: List[float], limit: int = 5) -> List[Dict]:
        """Search similar chunks using client-side Cosine Similarity."""
        # 1. Fetch all chunks (ID, content, page, embedding)
        # Check cache or fetch fresh? For now fetch fresh to be safe.
        query = "SELECT @rid, content, page_number, embedding FROM Chunk"
        res = self.query(query)
        
        if not res:
            return []

        import numpy as np
        
        # 2. Compute Cosine Similarity
        # Sim(A, B) = dot(A, B) / (norm(A) * norm(B))
        # Assuming embeddings are already normalized by the provider? Not safe to assume.
        
        query_vec = np.array(embedding)
        norm_q = np.linalg.norm(query_vec)
        if norm_q == 0:
            return []
            
        scored_chunks = []
        for r in res:
            vec = r.get("embedding")
            if not vec:
                continue
            
            # Ensure vec is list of floats
            if isinstance(vec, str):
                continue # Should not happen if LIST type used, but safety first
                
            doc_vec = np.array(vec)
            norm_d = np.linalg.norm(doc_vec)
            
            if norm_d == 0:
                score = 0
            else:
                score = np.dot(query_vec, doc_vec) / (norm_q * norm_d)
            
            r["distance"] = 1 - score # Convert similarity to distance-like (lower is better? or just use score)
            # Standard vector search often returns distance.
            # If we want Similarity (Higher is better), we should sort DESC.
            # If retrieval expects distance (Lower is better), we do 1 - score.
            # Let's use score and sort DESC.
            
            scored_chunks.append({"content": r.get("content"), "page_number": r.get("page_number"), "score": score})
            
        # 3. Sort and Limit
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:limit]

    def create_conversation(self) -> str:
        cmd = "INSERT INTO Conversation SET started_at = sysdate()"
        res = self.execute_command(cmd)
        if res and "result" in res:
            return res["result"][0].get("@rid")
        return ""

    def add_message(self, conv_rid: str, role: str, content: str):
        cmd = "INSERT INTO Message SET role = ?, content = ?, timestamp = sysdate()"
        payload = {"command": cmd, "params": {"0": role, "1": content}}
        url = f"{self.base_url}/command/{self.db_name}"
        res = requests.post(url, json=payload, auth=self.auth, headers=self.headers)
        msg_rid = None
        if res.status_code == 200:
             msg_rid = res.json().get("result", [])[0].get("@rid")
        
        if msg_rid and conv_rid:
            self.execute_command(f"CREATE EDGE HAS_MESSAGE FROM {conv_rid} TO {msg_rid}")
