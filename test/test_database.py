import pytest
import os
import sys
from datetime import datetime

# Ensure app modules are in path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.langchain.database import ArcadeDBClient

# Use a separate test database or mock?
# ArcadeDB doesn't easily support in-memory for this setup without docker args change.
# We will use a "test_study_tool" database.

TEST_DB_NAME = "test_study_tool"

@pytest.fixture(scope="module")
def db_client():
    client = ArcadeDBClient(db_name=TEST_DB_NAME)
    yield client
    # Teardown: drop test DB? 
    # client.execute_command(f"DROP DATABASE {TEST_DB_NAME}") 
    # Drop requires server-level command which client helper isn't quite built for directly in generic way
    # but let's leave it for inspection or manual cleanup for now, or assume CI resets it.
    pass

def test_connection(db_client):
    assert db_client.base_url
    # Ensure DB exists
    assert db_client._db_exists()

def test_schema_created(db_client):
    # Check if SourceFile type exists by querying it (should be empty but valid)
    query = "SELECT count(*) FROM SourceFile"
    res = db_client.query(query)
    assert len(res) == 1
    assert "count(*)" in res[0]

def test_insert_and_query_document(db_client):
    filename = f"test_doc_{int(datetime.now().timestamp())}.pdf"
    doc_type = "paper"
    
    # Insert
    rid = db_client.insert_document(filename, doc_type)
    assert rid.startswith("#")
    
    # Check existence
    assert db_client.document_exists(filename)
    
    # Query back
    res = db_client.query("SELECT FROM SourceFile WHERE filename = ?", params={"0": filename})
    assert len(res) > 0
    assert res[0]["filename"] == filename
    assert res[0]["type"] == doc_type

def test_insert_chunk_and_search(db_client):
    filename = "chunk_test.pdf"
    doc_rid = db_client.insert_document(filename, "slide")
    
    content = "This is a test chunk about quantum mechanics."
    embedding = [0.1] * 768 # Dummy embedding
    
    chunk_rid = db_client.insert_chunk(doc_rid, content, 1, embedding)
    # Note: insert_chunk implementation in database.py doesn't return RID explicitly?
    # Let's check the code or update if needed. 
    # The current implementation:
    # returns None? No, it does:
    # if r: chunk_rid = r[0].get("@rid") ... but function returns None implicitly?
    # Let's verify source. explicit return missing in `insert_chunk`?
    
    # Checking by query
    res = db_client.query("SELECT FROM Chunk WHERE content = ?", params={"0": content})
    assert len(res) > 0
    assert res[0]["content"] == content

def test_duplicate_document_check(db_client):
    filename = "dup_test.pdf"
    db_client.insert_document(filename, "textbook")
    assert db_client.document_exists(filename)
    # Should safely handle re-check
    assert db_client.document_exists(filename)
