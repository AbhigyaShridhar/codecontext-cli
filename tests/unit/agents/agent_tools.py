"""
Tests for agent tools
"""

import pytest
import os
from pathlib import Path

from codecontext.core.parser import CodebaseParser
from codecontext.search.vectorstore import MongoDBVectorStore
from codecontext.agents.tools import (
    initialize_tools,
    search_codebase,
    analyze_project_patterns,
    get_codebase_statistics,
    get_function_dependencies,
)


@pytest.fixture
def sample_project(tmp_path):
    """Create a sample project with realistic code"""
    # Create source files
    src = tmp_path / "src"
    src.mkdir()
    
    # Create auth module with login function
    (src / "auth.py").write_text("""
def login(username: str, password: str) -> bool:
    '''Authenticate user and create session'''
    if not validate_credentials(username, password):
        return False
    create_session(username)
    return True

def validate_credentials(username: str, password: str) -> bool:
    '''Validate user credentials against database'''
    # Check database
    return username and password

def create_session(username: str):
    '''Create user session with token'''
    pass

async def get_user_profile(user_id: int):
    '''Fetch user profile from API'''
    return {"id": user_id, "name": "User"}
""")
    
    # Create API module
    (src / "api.py").write_text("""
def handle_request(request):
    '''Handle incoming API request'''
    return {"status": "ok"}

class APIClient:
    '''Client for making API calls'''
    
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    def get(self, path: str):
        '''Make GET request'''
        pass
""")
    
    # Create requirements.txt
    (tmp_path / "requirements.txt").write_text("""
pytest==7.4.0
fastapi==0.100.0
requests==2.31.0
""")
    
    # Parse
    parser = CodebaseParser(tmp_path, ignore_patterns=[])
    codebase = parser.parse()
    
    return tmp_path, codebase


class TestToolInitialization:
    """Tests for tool initialization"""
    
    def test_initialize_tools(self, sample_project):
        """Test tool initialization"""
        
        # Skip if no credentials
        if not os.getenv("MONGODB_URI") or not os.getenv("OPENAI_API_KEY"):
            pytest.skip("Credentials required")
        
        project_root, codebase = sample_project
        
        # Create vector store
        store = MongoDBVectorStore(
            connection_string=os.getenv("MONGODB_URI"),
            database_name="codecontext_test",
            collection_name="tools_test",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        try:
            # Index
            store.index_codebase(codebase)
            
            initialize_tools(codebase, store, project_root)
            
            result = analyze_project_patterns.invoke({})
            
            assert "PROJECT PATTERNS" in result
            assert "pytest" in result.lower()
        
        finally:
            store.clear()
            store.close()


# tests/unit/agents/test_tools.py

class TestSearchTools:
    """Tests for search tools"""

    def test_search_codebase_tool(self, sample_project):
        """Test search_codebase tool"""

        if not os.getenv("MONGODB_URI") or not os.getenv("OPENAI_API_KEY"):
            pytest.skip("Credentials required")

        project_root, codebase = sample_project

        print(f"\n=== Test Setup ===")
        print(f"Project root: {project_root}")
        print(f"Total files: {codebase.total_files}")
        print(f"Total functions: {codebase.total_functions}")

        # List all functions
        all_funcs = [f.name for file in codebase.files.values() for f in file.get_all_functions()]
        print(f"Functions: {all_funcs}")

        store = MongoDBVectorStore(
            connection_string=os.getenv("MONGODB_URI"),
            database_name="codecontext_test",
            collection_name="search_test",
            api_key=os.getenv("OPENAI_API_KEY")
        )

        try:
            # Clear and index
            print("\n=== Indexing ===")
            store.clear()
            store.index_codebase(codebase)

            # Check stats
            stats = store.get_stats()
            print(f"Stats: {stats}")
            assert stats['total_items'] > 0, f"No items indexed! Stats: {stats}"

            # Initialize tools (IMPORTANT - this sets up the state)
            print("\n=== Initializing Tools ===")
            initialize_tools(codebase, store, project_root)

            # Try direct search first
            print("\n=== Direct Search ===")
            direct_results = store.search("user authentication login", k=5)
            print(f"Found {len(direct_results)} results")
            for r in direct_results[:3]:
                print(f"  - {r.name} (score: {r.score:.3f}) in {Path(r.file).name}")

            # Now try tool search
            print("\n=== Tool Search ===")
            result = search_codebase.invoke({
                "query": "user authentication login",
                "k": 5
            })

            print(f"Result:\n{result}")

            # Assertions
            assert "found 0 results" not in result.lower(), "Search should find results"
            assert "login" in result.lower(), "Should find login function"

        finally:
            store.clear()
            store.close()


class TestAnalysisTools:
    """Tests for analysis tools"""
    
    def test_get_function_dependencies(self, sample_project):
        """Test function dependency analysis"""
        
        if not os.getenv("MONGODB_URI") or not os.getenv("OPENAI_API_KEY"):
            pytest.skip("Credentials required")
        
        project_root, codebase = sample_project
        
        store = MongoDBVectorStore(
            connection_string=os.getenv("MONGODB_URI"),
            database_name="codecontext_test",
            collection_name="deps_test",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        try:
            # Index
            store.index_codebase(codebase)
            
            # Initialize tools
            initialize_tools(codebase, store, project_root)
            
            # Get dependencies of login function - use .invoke()
            result = get_function_dependencies.invoke({
                "file_path": "src/auth.py",
                "function_name": "login"
            })
            
            print(f"\nDependencies:\n{result}")
            
            # Should show what login() calls
            assert "validate_credentials" in result
            assert "create_session" in result
        
        finally:
            store.clear()
            store.close()
    
    def test_get_codebase_statistics(self, sample_project):
        """Test codebase statistics"""
        
        if not os.getenv("MONGODB_URI") or not os.getenv("OPENAI_API_KEY"):
            pytest.skip("Credentials required")
        
        project_root, codebase = sample_project
        
        store = MongoDBVectorStore(
            connection_string=os.getenv("MONGODB_URI"),
            database_name="codecontext_test",
            collection_name="stats_test",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        try:
            # Index
            store.index_codebase(codebase)
            
            # Initialize tools
            initialize_tools(codebase, store, project_root)
            
            # Get statistics - use .invoke()
            result = get_codebase_statistics.invoke({})
            
            print(f"\nStatistics:\n{result}")
            
            assert "CODEBASE STATISTICS" in result
            assert "Total functions:" in result
            assert "Total classes:" in result
        
        finally:
            store.clear()
            store.close()
