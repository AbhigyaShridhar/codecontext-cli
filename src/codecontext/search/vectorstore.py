from typing import List, Dict, Optional, Any
import certifi
import os
from dataclasses import dataclass

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import openai

from codecontext.core.codebase import Codebase, FunctionInfo, ClassInfo


# =============================================================================
# EMBEDDING GENERATOR
# =============================================================================

class EmbeddingGenerator:
    """
    Generate embeddings using OpenAI API

    Model: text-embedding-3-small
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize embedding generator

        Args:
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = "text-embedding-3-small"
        self.dimensions = 1536

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for text

        Args:
            text: Text to embed

        Returns:
            Embedding vector (1536 floats)

        Raises:
            openai.OpenAIError: If API call fails
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding

        except openai.OpenAIError as e:
            raise RuntimeError(f"Failed to generate embedding: {e}")

    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (more efficient)

        OpenAI allows up to 2048 texts per request, but we batch smaller
        for better error handling.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call

        Returns:
            List of embedding vectors
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

            except openai.OpenAIError as e:
                raise RuntimeError(f"Failed to generate embeddings for batch {i}: {e}")

        return all_embeddings


# =============================================================================
# SEARCH RESULT
# =============================================================================

@dataclass
class SearchResult:
    """Search result from vector store"""
    id: str
    type: str  # "function" or "class"
    name: str
    file: str
    score: float  # Similarity score (0-1, higher = more similar)
    metadata: Dict[str, Any]
    text: str  # Original searchable text


# =============================================================================
# MONGODB VECTOR STORE
# =============================================================================

class MongoDBVectorStore:
    """
    MongoDB Atlas vector store

    Stores code embeddings and also provide semantic search.

    Features:
    - Persistent cloud storage
    - Fast vector search
    - Rich metadata support
    - Production-ready
    """

    def __init__(
            self,
            connection_string: str,
            database_name: str = "codecontext",
            collection_name: str = "codebase",
            api_key: Optional[str] = None
    ):
        """
        Initialize MongoDB vector store

        Args:
            connection_string: MongoDB Atlas connection string
            database_name: Database name (default: codecontext)
            collection_name: Collection name (default: codebase)
            api_key: OpenAI API key for embeddings

        Raises:
            ConnectionFailure: If connection to MongoDB fails
        """
        self.connection_string = connection_string
        self.database_name = database_name
        self.collection_name = collection_name

        # Connect to MongoDB
        try:
            self.client = MongoClient(
                connection_string,
                tlsCAFile=certifi.where()
            )

            # Test connection
            self.client.admin.command('ping')

        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            raise ConnectionError(f"Failed to connect to MongoDB: {e}")

        # Get database and collection
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]

        # Initialize embeddings
        self.embedder = EmbeddingGenerator(api_key=api_key)

    def ensure_vector_index(self) -> bool:
        """
        Ensure vector search index exists, create if needed

        Returns:
            True if index exists or was created successfully
        """
        # Check if an index already exists
        print("Checking for existing vector search index...")
        if self._vector_index_exists():
            print("  ✓ Vector search index already exists")
            return True

        # Create the index
        print("Creating vector search index...")
        try:
            self._create_vector_index()

            # Verify it was created
            import time
            print("  Waiting 2 seconds for index to register...")
            time.sleep(2)

            if self._vector_index_exists():
                print("  ✓ Index verified")
                return True
            else:
                print("  ⚠ Index created but not yet visible (may take 1-2 minutes to build)")
                return True  # Still return True - index is being built

        except Exception as e:
            print(f"  ✗ Failed to create index: {e}")
            return False

    def _vector_index_exists(self) -> bool:
        """
        Check if vector search index exists

        Returns:
            True if 'vector_index' exists
        """
        try:
            indexes = list(self.collection.list_search_indexes())

            for idx in indexes:
                if idx.get("name") == "vector_index":
                    # Check if it's ready
                    status = idx.get("status", idx.get("queryable"))
                    if status in ["READY", True, "ACTIVE"]:
                        return True
                    else:
                        print(f"  Index exists but status is: {status}")
                        return False

            return False

        except AttributeError:
            # Method doesn't exist in this pymongo version
            pass
        except Exception as e:
            print(f"  list_search_indexes error: {e}")

        # Method 2: Command API
        try:
            result = self.db.command({
                "listSearchIndexes": self.collection_name
            })

            # Extract indexes from cursor
            indexes = result.get("cursor", {}).get("firstBatch", [])

            for idx in indexes:
                if idx.get("name") == "vector_index":
                    return True

            return False

        except Exception as e:
            print(f"  listSearchIndexes command error: {e}")

        # Cannot determine - assume doesn't exist
        return False

    def _ensure_collection_exists(self):
        """Create the collection if it doesn't exist yet."""
        if self.collection_name not in self.db.list_collection_names():
            self.db.create_collection(self.collection_name)

    def _create_vector_index(self):
        """
        Create vector search index

        Uses the same syntax that works in MongoDB Node.js driver.
        """
        self._ensure_collection_exists()

        index_model = {
            "name": "vector_index",
            "definition": {
                "mappings": {
                    "dynamic": True,
                    "fields": {
                        "embedding": {
                            "type": "knnVector",
                            "dimensions": 1536,
                            "similarity": "cosine"
                        }
                    }
                }
            }
        }

        try:
            result = self.collection.create_search_index(model=index_model)
            print(f"  ✓ Index created successfully (ID: {result})")
            return
        except AttributeError:
            # Method doesn't exist, try command API
            pass
        except Exception as e:
            # Log error but try other methods
            print(f"  create_search_index failed: {e}")

        # Fallback: Try command API
        try:
            result = self.db.command({
                "createSearchIndexes": self.collection_name,
                "indexes": [index_model]
            })

            if result.get("ok") == 1:
                print(f"  ✓ Index created via command API")
                return
            else:
                raise RuntimeError(f"Command returned: {result}")

        except Exception as e:
            raise RuntimeError(f"Failed to create index: {e}")

    def index_codebase(
            self,
            codebase: Codebase,
            batch_size: int = 100,
            progress_callback: Optional[callable] = None
    ):
        """
        Index the entire codebase into vector store

        This will:
        1. Ensure vector search index exists (create if needed)
        2. Clear existing data
        3. Extract all functions and classes
        4. Generate embeddings in batches
        5. Store in MongoDB

        Args:
            codebase: Parsed codebase
            batch_size: Embedding batch size
            progress_callback: Optional callback(current, total)
        """
        # Ensure vector index exists BEFORE indexing
        self.ensure_vector_index()

        # Clear existing data
        self.collection.delete_many({})

        # Prepare documents
        documents = []
        texts_to_embed = []

        # Index functions
        for file_info in codebase.files.values():
            for func in file_info.get_all_functions():
                # Build searchable text
                text = self._build_function_text(func)
                texts_to_embed.append(text)

                # Prepare the document (without embedding yet)
                documents.append({
                    "type": "function",
                    "name": func.name,
                    "file": str(func.file_path),
                    "line_start": func.line_start,
                    "line_end": func.line_end,
                    "signature": func.signature,
                    "docstring": func.docstring or "",
                    "code": func.code,
                    "complexity": func.cyclomatic_complexity,
                    "is_async": func.is_async,
                    "is_method": func.is_method,
                    "text": text,
                    # embedding will be added later
                })

        # Index classes
        for file_info in codebase.files.values():
            for cls in file_info.classes:
                # Build searchable text
                text = self._build_class_text(cls)
                texts_to_embed.append(text)

                # Prepare document
                documents.append({
                    "type": "class",
                    "name": cls.name,
                    "file": str(cls.file_path),
                    "line_start": cls.line_start,
                    "line_end": cls.line_end,
                    "docstring": cls.docstring or "",
                    "code": cls.code[:1000],  # Limit class code
                    "methods": [m.name for m in cls.methods],
                    "bases": cls.bases,
                    "text": text,
                })

        total = len(documents)

        if total == 0:
            return

        # Generate embeddings in batches
        all_embeddings = []
        for i in range(0, len(texts_to_embed), batch_size):
            batch_texts = texts_to_embed[i:i + batch_size]
            batch_embeddings = self.embedder.embed_batch(batch_texts, batch_size=batch_size)
            all_embeddings.extend(batch_embeddings)

            if progress_callback:
                progress_callback(min(i + batch_size, total), total)

        # Add embeddings to documents
        for doc, embedding in zip(documents, all_embeddings):
            doc["embedding"] = embedding

        # Insert into MongoDB
        if documents:
            self.collection.insert_many(documents)

        if progress_callback:
            progress_callback(total, total)

    def search(
            self,
            query: str,
            k: int = 10,
            filter_type: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search for code similar to query

        Uses MongoDB Atlas vector search ($vectorSearch aggregation).

        Args:
            query: Natural language query
            k: Number of results
            filter_type: Filter by "function" or "class"

        Returns:
            List of SearchResult objects, sorted by similarity
        """
        # Generate query embedding
        query_embedding = self.embedder.embed(query)

        # Build aggregation pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": min(k * 10, 100),
                    "limit": k
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "type": 1,
                    "name": 1,
                    "file": 1,
                    "text": 1,
                    "signature": 1,
                    "docstring": 1,
                    "code": 1,
                    "complexity": 1,
                    "methods": 1,
                    "is_async": 1,
                    "is_method": 1,
                    "bases": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        # Add type filter if specified
        if filter_type:
            pipeline.insert(1, {"$match": {"type": filter_type}})

        # Execute search
        try:
            cursor = self.collection.aggregate(pipeline)
            results = list(cursor)

        except Exception as e:
            error_msg = str(e).lower()

            # Check if it's an index issue
            if "index" in error_msg or "vector" in error_msg:
                # Try to create index automatically
                print("Vector search index not found. Attempting to create...")

                if self.ensure_vector_index():
                    print("Index created. Please wait 1-2 minutes for it to build, then try again.")
                    raise RuntimeError(
                        "Vector search index was just created. "
                        "Please wait 1-2 minutes for it to build, then retry your search."
                    )
                else:
                    # Could not create automatically
                    from codecontext.search.mongodb_setup import show_vector_index_instructions
                    print("\nAutomatic index creation failed.")
                    show_vector_index_instructions()
                    raise RuntimeError(
                        "Vector search index not found and could not be created automatically. "
                        "Please create it manually using the instructions above."
                    )

            # Some other error
            raise

        # Convert to SearchResult objects
        search_results = []
        for doc in results:
            search_results.append(SearchResult(
                id=str(doc["_id"]),
                type=doc["type"],
                name=doc["name"],
                file=doc["file"],
                score=doc["score"],
                metadata={
                    "signature": doc.get("signature", ""),
                    "docstring": doc.get("docstring", ""),
                    "complexity": doc.get("complexity", 0),
                    "methods": doc.get("methods", []),
                    "is_async": doc.get("is_async", False),
                    "is_method": doc.get("is_method", False),
                    "bases": doc.get("bases", []),
                },
                text=doc["text"]
            ))

        return search_results

    @staticmethod
    def _build_function_text(func: FunctionInfo) -> str:
        """Build searchable text for a function"""
        parts = [f"Function: {func.name}", f"Signature: {func.signature}"]

        # Function name (important!)

        # Signature (shows parameters and types)

        # Docstring (natural language description)
        if func.docstring:
            parts.append(f"Description: {func.docstring}")

        # Code preview (first 500 chars of implementation)
        if func.code:
            code_preview = func.code[:500]
            parts.append(f"Implementation: {code_preview}")

        # Function calls (what it depends on)
        if func.calls:
            parts.append(f"Calls: {', '.join(func.calls[:10])}")

        # Async indicator
        if func.is_async:
            parts.append("Type: async function")

        return "\n".join(parts)

    @staticmethod
    def _build_class_text(cls: ClassInfo) -> str:
        """Build searchable text for a class"""
        parts = [f"Class: {cls.name}"]

        # Class name

        # Docstring
        if cls.docstring:
            parts.append(f"Description: {cls.docstring}")

        # Methods (shows what the class can do)
        if cls.methods:
            method_names = [m.name for m in cls.methods[:10]]
            parts.append(f"Methods: {', '.join(method_names)}")

        # Inheritance
        if cls.bases:
            parts.append(f"Inherits from: {', '.join(cls.bases)}")

        return "\n".join(parts)

    def clear(self):
        """Clear all data from vector store"""
        self.collection.delete_many({})

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        total = self.collection.count_documents({})
        functions = self.collection.count_documents({"type": "function"})
        classes = self.collection.count_documents({"type": "class"})

        return {
            "total_items": total,
            "functions": functions,
            "classes": classes,
            "database": self.database_name,
            "collection": self.collection_name,
        }

    def close(self):
        """Close MongoDB connection"""
        self.client.close()
