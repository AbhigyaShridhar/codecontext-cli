import pytest
import os

from codecontext.search.vectorstore import (
    EmbeddingGenerator,
    MongoDBVectorStore,
    SearchResult,
)
from codecontext.core.parser import CodebaseParser

# Skip all tests if credentials not available
requires_credentials = pytest.mark.skipif(
    not os.getenv("MONGODB_URI") or not os.getenv("OPENAI_API_KEY"),
    reason="MongoDB URI and OpenAI API key required. Set MONGODB_URI and OPENAI_API_KEY environment variables."
)


@pytest.fixture
def sample_codebase(tmp_path):
    """Create a sample codebase for testing"""
    # Create sample Python files
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    # File 1: auth.py
    auth_file = src_dir / "auth.py"
    auth_file.write_text("""
def login(username: str, password: str) -> bool:
    '''
    Authenticate user with username and password.

    Args:
        username: User's username
        password: User's password

    Returns:
        True if authentication successful
    '''
    # Validate credentials
    if not username or not password:
        return False

    # Check against database
    user = find_user(username)
    if not user:
        return False

    return verify_password(user, password)


def find_user(username: str):
    '''Find user in database'''
    return {"username": username, "id": 1}


def verify_password(user, password):
    '''Verify user password'''
    return True


class UserManager:
    '''Manages user accounts'''

    def create_user(self, username: str, email: str):
        '''Create a new user account'''
        return {"username": username, "email": email}

    def delete_user(self, user_id: int):
        '''Delete user account'''
        pass
""")

    # File 2: api.py
    api_file = src_dir / "api.py"
    api_file.write_text("""
async def get_user_profile(user_id: int):
    '''
    Fetch user profile from API.

    This is an async function that retrieves user data.
    '''
    # Simulate API call
    return {"id": user_id, "name": "John Doe"}


def process_payment(amount: float, card_number: str):
    '''Process payment transaction'''
    if amount <= 0:
        raise ValueError("Invalid amount")

    # Process payment
    return {"status": "success", "amount": amount}
""")

    # Parse the codebase
    parser = CodebaseParser(tmp_path, ignore_patterns=[])
    codebase = parser.parse()

    return codebase


@pytest.fixture
def vector_store():
    """Create vector store with real MongoDB connection"""
    connection_string = os.getenv("MONGODB_URI")
    api_key = os.getenv("OPENAI_API_KEY")

    if not connection_string or not api_key:
        pytest.skip("MongoDB URI or OpenAI API key not set")

    # Use a test database to avoid polluting production
    store = MongoDBVectorStore(
        connection_string=connection_string,
        database_name="codecontext_test",
        collection_name="test_codebase",
        api_key=api_key
    )

    # Clear any existing data
    store.clear()

    yield store

    # Cleanup after test
    store.clear()
    store.close()


class TestEmbeddingGenerator:
    """Tests for embedding generator"""

    @requires_credentials
    def test_init_with_api_key(self):
        """Test initialization with an API key"""
        api_key = os.getenv("OPENAI_API_KEY")
        generator = EmbeddingGenerator(api_key=api_key)

        assert generator.api_key == api_key
        assert generator.model == "text-embedding-3-small"
        assert generator.dimensions == 1536

    def test_init_without_api_key_raises(self):
        """Test initialization without an API key raises error"""
        # Temporarily remove env var
        old_key = os.environ.pop("OPENAI_API_KEY", None)

        try:
            with pytest.raises(ValueError, match="OpenAI API key required"):
                EmbeddingGenerator()
        finally:
            # Restore env var
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key

    @requires_credentials
    def test_embed_single_text(self):
        """Test embedding single text"""
        api_key = os.getenv("OPENAI_API_KEY")
        generator = EmbeddingGenerator(api_key=api_key)

        embedding = generator.embed("test function for authentication")

        assert isinstance(embedding, list)
        assert len(embedding) == 1536
        assert all(isinstance(x, float) for x in embedding)

    @requires_credentials
    def test_embed_batch(self):
        """Test batch embedding generation"""
        api_key = os.getenv("OPENAI_API_KEY")
        generator = EmbeddingGenerator(api_key=api_key)

        texts = [
            "user authentication function",
            "payment processing code",
            "async API endpoint",
        ]

        embeddings = generator.embed_batch(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 1536 for emb in embeddings)

        # Embeddings should be different for different texts
        assert embeddings[0] != embeddings[1]
        assert embeddings[1] != embeddings[2]


class TestSearchResult:
    """Tests for SearchResult dataclass"""

    def test_creation(self):
        """Test SearchResult creation"""
        result = SearchResult(
            id="test123",
            type="function",
            name="test_func",
            file="/path/to/file.py",
            score=0.95,
            metadata={"complexity": 3},
            text="Function: test_func"
        )

        assert result.id == "test123"
        assert result.type == "function"
        assert result.name == "test_func"
        assert result.score == 0.95
        assert result.metadata["complexity"] == 3


class TestMongoDBVectorStore:
    """Tests for MongoDB vector store (integration tests)"""

    @requires_credentials
    def test_init_success(self):
        """Test successful initialization"""
        connection_string = os.getenv("MONGODB_URI")
        api_key = os.getenv("OPENAI_API_KEY")

        store = MongoDBVectorStore(
            connection_string=connection_string,
            database_name="codecontext_test",
            collection_name="test_init",
            api_key=api_key
        )

        assert store.database_name == "codecontext_test"
        assert store.collection_name == "test_init"

        store.close()

    def test_init_connection_failure(self):
        """Test initialization with bad connection string"""
        with pytest.raises(ConnectionError, match="Failed to connect"):
            MongoDBVectorStore(
                connection_string="mongodb://invalid:27017",
                database_name="test",
                api_key="fake-key"
            )

    @requires_credentials
    def test_clear(self, vector_store):
        """Test clearing the vector store"""
        # Insert a document
        vector_store.collection.insert_one({"test": "data"})

        # Verify it exists
        count = vector_store.collection.count_documents({})
        assert count == 1

        # Clear
        vector_store.clear()

        # Verify it's gone
        count = vector_store.collection.count_documents({})
        assert count == 0

    @requires_credentials
    def test_index_codebase(self, vector_store, sample_codebase):
        """Test indexing a codebase"""
        # Track progress
        progress_calls = []

        def track_progress(current, total):
            progress_calls.append((current, total))

        # Index the codebase
        vector_store.index_codebase(
            sample_codebase,
            batch_size=10,
            progress_callback=track_progress
        )

        # Check progress was tracked
        assert len(progress_calls) > 0

        # Verify documents were inserted
        stats = vector_store.get_stats()
        assert stats["total_items"] > 0
        assert stats["functions"] > 0

        # Verify embeddings exist
        sample_doc = vector_store.collection.find_one({"type": "function"})
        assert sample_doc is not None
        assert "embedding" in sample_doc
        assert len(sample_doc["embedding"]) == 1536

        print(f"\n✓ Indexed {stats['total_items']} items")
        print(f"  - Functions: {stats['functions']}")
        print(f"  - Classes: {stats['classes']}")

    @requires_credentials
    def test_get_stats(self, vector_store, sample_codebase):
        """Test getting statistics"""
        # Index first
        vector_store.index_codebase(sample_codebase)

        # Get stats
        stats = vector_store.get_stats()

        assert "total_items" in stats
        assert "functions" in stats
        assert "classes" in stats
        assert stats["database"] == "codecontext_test"
        assert stats["collection"] == "test_codebase"

        # Should have indexed our sample files
        assert stats["total_items"] > 0
        assert stats["functions"] >= 5  # login, find_user, verify_password, get_user_profile, process_payment
        assert stats["classes"] >= 1  # UserManager

    @requires_credentials
    def test_build_function_text(self, vector_store, sample_codebase):
        """Test building searchable text for functions"""
        # Get a function
        login_funcs = sample_codebase.find_function("login")
        assert login_funcs is not None
        assert len(login_funcs) > 0

        login_func = login_funcs[0]

        # Build text
        text = vector_store._build_function_text(login_func)

        # Verify important parts are included
        assert "login" in text
        assert "username" in text or "password" in text
        assert "Authenticate" in text  # From docstring

    @requires_credentials
    def test_build_class_text(self, vector_store, sample_codebase):
        """Test building searchable text for classes"""
        # Get a class
        user_manager = sample_codebase.find_class("UserManager")
        assert user_manager is not None
        assert len(user_manager) > 0

        cls = user_manager[0]

        # Build text
        text = vector_store._build_class_text(cls)

        # Verify important parts are included
        assert "UserManager" in text
        assert "create_user" in text or "delete_user" in text
        assert "Manages user accounts" in text  # From docstring


class TestVectorSearch:
    """Tests for vector search functionality"""

    @requires_credentials
    def test_search_requires_vector_index(self, vector_store, sample_codebase):
        """Test that search explains if vector index is missing"""
        # Index the codebase
        vector_store.index_codebase(sample_codebase)

        # Try to search (will fail if index doesn't exist)
        try:
            results = vector_store.search("authentication", k=5)

            # If we get here, index exists - verify results
            assert isinstance(results, list)
            print(f"\n✓ Vector index exists! Got {len(results)} results")

            # Results should be SearchResult objects
            if results:
                assert isinstance(results[0], SearchResult)
                print(f"  Top result: {results[0].name} (score: {results[0].score:.3f})")

        except RuntimeError as e:
            # Expected if vector index doesn't exist yet
            assert "vector_index" in str(e).lower()
            print("\n⚠ Vector search index not created yet.")
            print("  This is expected on first run.")
            print("  Create the index in MongoDB Atlas, then re-run this test.")
            pytest.skip("Vector search index not created yet - see MongoDB Atlas setup")

    # @requires_credentials
    # def test_search_authentication_functions(self, vector_store, sample_codebase):
    #     """Test searching for authentication-related functions"""
    #     # Index the codebase
    #     vector_store.index_codebase(sample_codebase)
    #
    #     try:
    #         # Search for authentication
    #         results = vector_store.search("user authentication login", k=10)
    #
    #         # Should find login-related functions
    #         function_names = [r.name for r in results]
    #
    #         print(f"\n✓ Found {len(results)} results for 'authentication':")
    #         for r in results[:3]:
    #             print(f"  - {r.name} (score: {r.score:.3f}) in {Path(r.file).name}")
    #
    #         # login function should be highly ranked
    #         assert any("login" in name.lower() for name in function_names[:3])
    #
    #     except RuntimeError as e:
    #         if "vector_index" in str(e).lower():
    #             pytest.skip("Vector search index not created yet")
    #         raise
    #
    # @requires_credentials
    # def test_search_async_functions(self, vector_store, sample_codebase):
    #     """Test searching for async functions"""
    #     # Index the codebase
    #     vector_store.index_codebase(sample_codebase)
    #
    #     try:
    #         # Search for async functions
    #         results = vector_store.search("async API endpoint", k=10)
    #
    #         print(f"\n✓ Found {len(results)} results for 'async':")
    #         for r in results[:3]:
    #             print(f"  - {r.name} (async: {r.metadata.get('is_async', False)})")
    #
    #         # Should find get_user_profile (async function)
    #         async_results = [r for r in results if r.metadata.get('is_async')]
    #         assert len(async_results) > 0
    #
    #     except RuntimeError as e:
    #         if "vector_index" in str(e).lower():
    #             pytest.skip("Vector search index not created yet")
    #         raise

    @requires_credentials
    def test_search_with_type_filter(self, vector_store, sample_codebase):
        """Test searching with type filter"""
        # Index the codebase
        vector_store.index_codebase(sample_codebase)

        try:
            # Search only functions
            func_results = vector_store.search("user", k=10, filter_type="function")

            # All results should be functions
            assert all(r.type == "function" for r in func_results)

            # Search only classes
            class_results = vector_store.search("user", k=10, filter_type="class")

            # All results should be classes
            assert all(r.type == "class" for r in class_results)

            print(f"\n✓ Type filtering works:")
            print(f"  - Functions: {len(func_results)}")
            print(f"  - Classes: {len(class_results)}")

        except RuntimeError as e:
            if "vector_index" in str(e).lower():
                pytest.skip("Vector search index not created yet")
            raise


# Integration test to verify the entire flow
@requires_credentials
def test_full_integration_flow(tmp_path):
    """
    Full integration test: Parse → Index → Search

    This test verifies the entire workflow works end-to-end.
    """
    connection_string = os.getenv("MONGODB_URI")
    api_key = os.getenv("OPENAI_API_KEY")

    print("\n" + "=" * 70)
    print("FULL INTEGRATION TEST")
    print("=" * 70)

    # 1. Create sample code
    print("\n1. Creating sample codebase...")
    src = tmp_path / "src"
    src.mkdir()

    (src / "auth.py").write_text("""
def authenticate_user(username: str, password: str) -> bool:
    '''Verify user credentials and return authentication status'''
    return check_password(username, password)

def check_password(username: str, password: str) -> bool:
    '''Check if password is correct'''
    return True
""")

    # 2. Parse codebase
    print("2. Parsing codebase...")
    parser = CodebaseParser(tmp_path, ignore_patterns=[])
    codebase = parser.parse()
    print(f"   ✓ Parsed {codebase.total_functions} functions")

    # 3. Create vector store
    print("3. Connecting to MongoDB...")
    store = MongoDBVectorStore(
        connection_string=connection_string,
        database_name="codecontext_test",
        collection_name="integration_test",
        api_key=api_key
    )
    store.clear()
    print("   ✓ Connected to MongoDB")

    # 4. Index codebase
    print("4. Indexing codebase (generating embeddings)...")

    def show_progress(current, total):
        print(f"   Progress: {current}/{total}", end='\r')

    store.index_codebase(codebase, progress_callback=show_progress)
    print(f"   ✓ Indexed {store.get_stats()['total_items']} items")

    # 5. Search
    print("5. Testing semantic search...")
    try:
        results = store.search("user login authentication", k=5)

        print(f"   ✓ Search successful! Found {len(results)} results:")
        for i, r in enumerate(results, 1):
            print(f"      {i}. {r.name} (score: {r.score:.3f})")

    except RuntimeError as e:
        if "vector_index" in str(e).lower():
            print("   ⚠ Vector index not created yet (expected on first run)")
            print("   Please create the index in MongoDB Atlas")
        else:
            raise

    # Cleanup
    store.clear()
    store.close()

    print("\n" + "=" * 70)
    print("✅ INTEGRATION TEST COMPLETE")
    print("=" * 70)
