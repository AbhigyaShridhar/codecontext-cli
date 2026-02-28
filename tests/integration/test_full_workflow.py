"""
Full workflow integration tests

Tests the complete user journey:
1. Initialize config
2. Parse codebase
3. Index into vector store
4. Search for code
5. Generate tests
"""

import pytest
import os
import tempfile
from pathlib import Path

from codecontext.core.parser import CodebaseParser
from codecontext.search.vectorstore import MongoDBVectorStore
from codecontext.agents.base import TestGenerationAgent
from codecontext.agents.tools import initialize_tools

# Skip all tests if credentials not available
pytestmark = pytest.mark.skipif(
    not os.getenv("MONGODB_URI") or not os.getenv("OPENAI_API_KEY") or not os.getenv("ANTHROPIC_API_KEY"),
    reason="Requires MongoDB, OpenAI, and Anthropic credentials"
)


@pytest.fixture
def sample_project():
    """Create a realistic sample project"""
    with tempfile.TemporaryDirectory() as tmp:
        project_root = Path(tmp)

        # Create project structure
        src = project_root / "src"
        src.mkdir()

        tests = project_root / "tests"
        tests.mkdir()

        # Create the main application code
        (src / "auth.py").write_text("""
'''User authentication module'''

from typing import Optional
import hashlib


def hash_password(password: str) -> str:
    '''Hash a password using SHA256'''
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, hashed: str) -> bool:
    '''Verify a password against its hash'''
    return hash_password(password) == hashed


def login(username: str, password: str) -> Optional[dict]:
    '''
    Authenticate a user and return user data

    Args:
        username: Username to authenticate
        password: Password to verify

    Returns:
        User dict if successful, None otherwise
    '''
    # Mock user database
    users = {
        "alice": {
            "username": "alice",
            "password_hash": hash_password("secret123"),
            "email": "alice@example.com"
        }
    }

    user = users.get(username)
    if not user:
        return None

    if verify_password(password, user["password_hash"]):
        return {"username": user["username"], "email": user["email"]}

    return None


def create_session(user_data: dict) -> str:
    '''Create a session token for authenticated user'''
    import uuid
    return str(uuid.uuid4())
""")

        (src / "api.py").write_text("""
'''API endpoints'''

from typing import Dict, Any
import json


async def handle_login(username: str, password: str) -> Dict[str, Any]:
    '''
    Handle login API request

    Args:
        username: Username
        password: Password

    Returns:
        API response with token or error
    '''
    from .auth import login, create_session

    user = login(username, password)

    if user:
        token = create_session(user)
        return {
            "success": True,
            "token": token,
            "user": user
        }

    return {
        "success": False,
        "error": "Invalid credentials"
    }


async def handle_logout(token: str) -> Dict[str, Any]:
    '''Handle logout API request'''
    # Invalidate token
    return {"success": True}
""")

        (src / "database.py").write_text("""
'''Database utilities'''

import sqlite3
from typing import Optional, List, Dict, Any


class Database:
    '''SQLite database wrapper'''

    def __init__(self, db_path: str):
        '''Initialize database connection'''
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self):
        '''Connect to database'''
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        '''Execute a query'''
        if not self.conn:
            self.connect()
        return self.conn.execute(query, params)

    def fetch_one(self, query: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        '''Fetch one result'''
        cursor = self.execute(query, params)
        row = cursor.fetchone()
        return dict(row) if row else None

    def fetch_all(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        '''Fetch all results'''
        cursor = self.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def close(self):
        '''Close database connection'''
        if self.conn:
            self.conn.close()
""")

        # Create existing test file (to detect patterns from)
        (tests / "test_auth.py").write_text("""
import pytest
from src.auth import hash_password, verify_password, login


def test_hash_password():
    '''Test password hashing'''
    hashed = hash_password("password123")
    assert len(hashed) == 64  # SHA256 produces 64 hex chars


def test_verify_password():
    '''Test password verification'''
    password = "test123"
    hashed = hash_password(password)

    assert verify_password(password, hashed)
    assert not verify_password("wrong", hashed)


def test_login_success():
    '''Test successful login'''
    result = login("alice", "secret123")

    assert result is not None
    assert result["username"] == "alice"
    assert result["email"] == "alice@example.com"


def test_login_invalid_username():
    '''Test login with invalid username'''
    result = login("bob", "password")
    assert result is None


def test_login_invalid_password():
    '''Test login with invalid password'''
    result = login("alice", "wrong")
    assert result is None
""")

        # Create requirements.txt
        (project_root / "requirements.txt").write_text("""
pytest==7.4.0
pytest-asyncio==0.21.0
""")

        # Create pyproject.toml
        (project_root / "pyproject.toml").write_text("""
[project]
name = "test-app"
version = "0.1.0"

[project.optional-dependencies]
test = ["pytest>=7.0", "pytest-asyncio"]
""")

        yield project_root


class TestFullWorkflow:
    """Test the complete workflow"""

    def test_parse_index_search_workflow(self, sample_project):
        """Test: Parse → Index → Search"""

        # 1. Parse codebase
        parser = CodebaseParser(
            root=sample_project,
            ignore_patterns=["__pycache__", ".git"]
        )
        codebase = parser.parse()

        assert codebase.total_files > 0
        assert codebase.total_functions > 0

        print(f"\n✓ Parsed {codebase.total_functions} functions from {codebase.total_files} files")

        # 2. Connect to vector store
        store = MongoDBVectorStore(
            connection_string=os.getenv("MONGODB_URI"),
            database_name="codecontext_integration_test",
            collection_name="workflow_test",
            api_key=os.getenv("OPENAI_API_KEY")
        )

        try:
            # 3. Index codebase
            store.clear()
            store.index_codebase(codebase)

            stats = store.get_stats()
            assert stats['total_items'] > 0

            print(f"✓ Indexed {stats['total_items']} items")

            # 4. Search for authentication functions
            results = store.search("user authentication and login", k=5)

            assert len(results) > 0
            print(f"✓ Found {len(results)} results for 'authentication'")

            # Verify we found the login function
            function_names = [r.name for r in results]
            assert "login" in function_names or any("login" in name.lower() for name in function_names)

            print(f"  Functions found: {function_names[:3]}")

            # 5. Search for database functions
            db_results = store.search("database query and fetch", k=5)
            assert len(db_results) > 0

            print(f"✓ Found {len(db_results)} results for 'database'")

        finally:
            store.clear()
            store.close()

    def test_agent_tools_workflow(self, sample_project):
        """Test: Tools → Agent → Generation"""

        # 1. Parse and index
        parser = CodebaseParser(sample_project, ignore_patterns=[])
        codebase = parser.parse()

        store = MongoDBVectorStore(
            connection_string=os.getenv("MONGODB_URI"),
            database_name="codecontext_integration_test",
            collection_name="agent_test",
            api_key=os.getenv("OPENAI_API_KEY")
        )

        try:
            store.clear()
            store.index_codebase(codebase)

            # 2. Initialize tools
            initialize_tools(codebase, store, sample_project)

            print("\n✓ Tools initialized")

            # 3. Create agent
            agent = TestGenerationAgent(
                codebase=codebase,
                vector_store=store,
                project_root=sample_project,
                llm_config={
                    "model": "claude-sonnet-4-20250514",
                    "api_key": os.getenv("ANTHROPIC_API_KEY"),
                    "temperature": 0.0,
                }
            )

            print("✓ Agent created")

            # 4. Generate test for a function
            print("\n✓ Generating test for 'handle_login' function...")

            test_code = agent.generate_test(
                target_function="handle_login",
                target_file="src/api.py"
            )

            assert test_code is not None
            assert len(test_code) > 0
            assert "def test_" in test_code
            assert "handle_login" in test_code

            print(f"✓ Generated {len(test_code)} characters of test code")
            print(f"\nGenerated test preview:")
            print(test_code[:500] + "...")

        finally:
            store.clear()
            store.close()

    def test_end_to_end_test_generation(self, sample_project):
        """Test: Complete end-to-end test generation"""

        # Parse
        parser = CodebaseParser(sample_project, ignore_patterns=[])
        codebase = parser.parse()

        # Index
        store = MongoDBVectorStore(
            connection_string=os.getenv("MONGODB_URI"),
            database_name="codecontext_integration_test",
            collection_name="e2e_test",
            api_key=os.getenv("OPENAI_API_KEY")
        )

        try:
            store.index_codebase(codebase)

            # Initialize tools
            initialize_tools(codebase, store, sample_project)

            # Create agent
            agent = TestGenerationAgent(
                codebase=codebase,
                vector_store=store,
                project_root=sample_project,
                llm_config={
                    "model": "claude-sonnet-4-20250514",
                    "api_key": os.getenv("ANTHROPIC_API_KEY"),
                    "temperature": 0.0,
                }
            )

            # Generate tests for multiple functions
            functions_to_test = [
                ("verify_password", "src/auth.py"),
                ("Database.fetch_one", "src/database.py"),
            ]

            for func_name, file_path in functions_to_test:
                print(f"\n✓ Generating test for {func_name}...")

                test_code = agent.generate_test(
                    target_function=func_name,
                    target_file=file_path
                )

                assert test_code is not None
                assert len(test_code) > 100
                assert "def test_" in test_code

                # Save to file
                test_file = sample_project / "tests" / f"test_generated_{func_name.replace('.', '_')}.py"
                test_file.write_text(test_code)

                print(f"  ✓ Saved to {test_file.name}")
                print(f"  ✓ {len(test_code)} characters")

        finally:
            store.clear()
            store.close()
