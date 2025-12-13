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
        """Check if database exists."""
        try:
            response = requests.get(
                f"{self.base_url}/server",
                json={"command": f"exists database {self.db_name}"},
                auth=self.auth,
                headers=self.headers
            )
            # The API behavior for 'exists' might vary, checking list of DBs is safer often, 
            # but let's try a simple command or check if we can connect.
            # actually server command 'exists database' might not be standard HTTP API.
            # Let's try querying the DB directly, if 404/401 it might not exist or need creation.
            # Alternative: Get list of databases
            res = requests.get(f"{self.base_url}/server/databases", auth=self.auth)
            if res.status_code == 200:
                dbs = res.json().get("result", [])
                return self.db_name in dbs
            return False
        except Exception as e:
            print(f"Error checking DB existence: {e}")
            return False

    def _init_schema(self):
        """Initialize Vertices and Edges."""
        # Define schema commands
        # Note: IF NOT EXISTS might not be supported for all commands in this version's SQL parser
        # We will attempt creation and ignore errors for now (simulating if not exists)
        commands = [
            "CREATE TYPE Document IF NOT EXISTS",
            "CREATE PROPERTY Document.filename STRING", 
            "CREATE PROPERTY Document.type STRING", 
            "CREATE PROPERTY Document.upload_date DATETIME",
            
            "CREATE TYPE Chunk IF NOT EXISTS",
            "CREATE PROPERTY Chunk.content STRING",
            "CREATE PROPERTY Chunk.page_number INTEGER",
            "CREATE PROPERTY Chunk.embedding LIST", # FLOAT_LIST might be LIST in simplified SQL
            
            # Index syntax: CREATE INDEX <name> ON <type> (<prop>) ...
            "CREATE INDEX Chunk_embedding ON Chunk (embedding) VECTOR NOTUNIQUE",
            
            "CREATE TYPE HAS_CHUNK IF NOT EXISTS", # Edge type matching naming convention? 
            # In ArcadeDB, Edge types are just Types extending E. 
            # Use: CREATE EXTENSION
            # Or just CREATE TYPE HAS_CHUNK IF NOT EXISTS EXTENDS E
            "CREATE TYPE HAS_CHUNK IF NOT EXISTS EXTENDS E",

            "CREATE TYPE Conversation IF NOT EXISTS",
            "CREATE PROPERTY Conversation.started_at DATETIME",
            
            "CREATE TYPE Message IF NOT EXISTS",
            "CREATE PROPERTY Message.role STRING",
            "CREATE PROPERTY Message.content STRING",
            "CREATE PROPERTY Message.timestamp DATETIME",
            
            "CREATE TYPE HAS_MESSAGE IF NOT EXISTS EXTENDS E"
        ]
        
        for cmd in commands:
            self.execute_command(cmd, language="sqlscript")

    def execute_command(self, command: str, language: str = "sql") -> Dict:
        url = f"{self.base_url}/command/{self.db_name}"
        payload = {"command": command, "language": language}
        response = requests.post(url, json=payload, auth=self.auth, headers=self.headers)
        if response.status_code != 200:
            print(f"Command failed: {command} -> {response.text}")
            return {}
        return response.json()

    def query(self, query: str, language: str = "sql") -> List[Dict]:
        url = f"{self.base_url}/query/{self.db_name}"
        payload = {"command": query, "language": language}
        response = requests.post(url, json=payload, auth=self.auth, headers=self.headers)
        if response.status_code != 200:
             print(f"Query failed: {query} -> {response.text}")
             return []
        return response.json().get("result", [])

    def insert_document(self, filename: str, doc_type: str) -> str:
        """Insert document metadata and return RID."""
        cmd = "INSERT INTO Document SET filename = ?, type = ?, upload_date = sysdate()"
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
        return ""

    def insert_chunk(self, doc_rid: str, content: str, page_num: int, embedding: List[float]):
        """Insert chunk and link to document."""
        # 1. Insert Chunk
        cmd_chunk = "INSERT INTO Chunk SET content = ?, page_number = ?, embedding = ?"
        payload_chunk = {
            "command": cmd_chunk,
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
        
        # 2. Link
        if chunk_rid and doc_rid:
            cmd_link = f"CREATE EDGE HAS_CHUNK FROM {doc_rid} TO {chunk_rid}"
            self.execute_command(cmd_link)
            
    def similarity_search(self, embedding: List[float], limit: int = 5) -> List[Dict]:
        """Search similar chunks."""
        # Syntax: SELECT FROM Chunk WHERE embedding VECTOR_NEIGHBORS [?, ?]
        # Or with index: SELECT FROM Chunk WHERE embedding MATCHES_VECTOR [?, ?] (depends on exact ArcadeDB/Lucene syntax)
        # Standard ArcadeDB 24.x Lucene syntax: 
        # v24.10: SELECT FROM Chunk WHERE vector_knn(embedding, ?) < ? (Maybe?)
        # Let's use the explicit index syntax if possible, or the function.
        # Assuming: SELECT FROM Chunk ORDER BY vector_distance(embedding, ?) ASC LIMIT ?
        # Note: vector_distance might be 'vec_distance' or similar. 
        # Let's try the dedicated vector search syntax: 
        # SELECT FROM index:Chunk_embedding WHERE key BETWEEN [...] -- this is for range
        # Official docs say: `SELECT vector_distance(embedding, ?) as distance, * FROM Chunk ORDER BY distance ASC LIMIT ?` is standard pattern.
        
        query = "SELECT content, page_number, vector_distance(embedding, ?) as distance FROM Chunk ORDER BY distance ASC LIMIT ?"
        payload = {
            "command": query,
            "params": {
                "0": embedding,
                "1": limit
            }
        }
        url = f"{self.base_url}/query/{self.db_name}"
        response = requests.post(url, json=payload, auth=self.auth, headers=self.headers)
        if response.status_code == 200:
            return response.json().get("result", [])
        return []

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
