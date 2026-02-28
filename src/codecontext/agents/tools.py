"""
LangChain tools for CodeContext agent

These tools give the agent access to:
- Semantic code search (vector store)
- Project pattern analysis (venv, dependencies, test patterns)
- Function dependency analysis
- Complexity metrics
"""

from typing import Optional, List, Dict
from langchain_core.tools import tool
from pathlib import Path
from dataclasses import is_dataclass

from codecontext.core.codebase import Codebase
from codecontext.search.vectorstore import MongoDBVectorStore
from codecontext.analysis.venv_scanner import VirtualEnvScanner
from codecontext.analysis.dependencies import parse_dependencies
from codecontext.analysis.test_patterns import detect_test_patterns


# =============================================================================
# TOOL STATE (shared across tools)
# =============================================================================

class ToolState:
    """
    Shared state for all tools

    This allows tools to access the parsed codebase, vector store, etc.
    without passing them as parameters (which LangChain doesn't support well).
    """
    codebase: Optional[Codebase] = None
    vector_store: Optional[MongoDBVectorStore] = None
    project_root: Optional[Path] = None
    venv_packages: Optional[Dict] = None
    dependencies: Optional[Dict] = None
    test_patterns: Optional[Dict] = None


# Global state instance
_state = ToolState()


def initialize_tools(
        codebase: Codebase,
        vector_store: MongoDBVectorStore,
        project_root: Path
):
    """
    Initialize tool state

    Call this before using the tools.

    Args:
        codebase: Parsed codebase
        vector_store: Vector store for semantic search
        project_root: Project root directory
    """
    _state.codebase = codebase
    _state.vector_store = vector_store
    _state.project_root = project_root

    # Analyze project patterns (cached)
    print("Analyzing project patterns...")

    # Scan virtual environment
    scanner = VirtualEnvScanner()
    _state.venv_packages = scanner.scan(limit=50)  # Top 50 packages

    # Parse dependencies
    _state.dependencies = parse_dependencies(project_root)

    # Detect test patterns
    _state.test_patterns = detect_test_patterns(codebase)

    print("✓ Tools initialized")


# =============================================================================
# SEARCH TOOLS
# =============================================================================

@tool
def search_codebase(query: str, k: int = 10, filter_type: Optional[str] = None) -> str:
    """
    Search the codebase using semantic similarity.

    Use this when you need to find relevant code based on natural language.

    Args:
        query: Natural language search query (e.g., "user authentication functions")
        k: Number of results to return (default: 10)
        filter_type: Optional filter - "function" or "class"

    Returns:
        JSON string with search results

    Examples:
        search_codebase("database connection handling")
        search_codebase("API endpoints for user management", filter_type="function")
        search_codebase("test fixtures for authentication", k=5)
    """
    if not _state.vector_store:
        return "Error: Vector store not initialized"

    try:
        results = _state.vector_store.search(query, k=k, filter_type=filter_type)

        # Format results as readable text
        output = [f"Found {len(results)} results for '{query}':\n"]

        for i, result in enumerate(results, 1):
            output.append(f"{i}. {result.name} ({result.type})")
            output.append(f"   File: {result.file}")
            output.append(f"   Similarity: {result.score:.3f}")

            if result.metadata.get("signature"):
                output.append(f"   Signature: {result.metadata['signature']}")

            if result.metadata.get("docstring"):
                doc = result.metadata["docstring"][:200]
                output.append(f"   Docstring: {doc}...")

            output.append("")

        return "\n".join(output)

    except Exception as e:
        return f"Error during search: {e}"


@tool
def find_package_usage_examples(package_name: str, k: int = 5) -> str:
    """
    Find examples of how a specific package is used in the codebase.

    Use this when you need to understand how a library/package is being used.

    Args:
        package_name: Name of the package (e.g., "fastapi", "pytest", "sqlalchemy")
        k: Number of examples to return

    Returns:
        Examples of package usage

    Examples:
        find_package_usage_examples("fastapi")
        find_package_usage_examples("pytest", k=3)
    """
    if not _state.vector_store:
        return "Error: Vector store not initialized"

    # Search for usage of this package
    query = f"usage of {package_name} library code examples"

    try:
        results = _state.vector_store.search(query, k=k)

        output = [f"Found {len(results)} examples using '{package_name}':\n"]

        for i, result in enumerate(results, 1):
            output.append(f"{i}. {result.name} in {Path(result.file).name}")

            # Show code snippet
            if result.metadata.get("signature"):
                output.append(f"   {result.metadata['signature']}")

            output.append("")

        return "\n".join(output)

    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# PROJECT ANALYSIS TOOLS
# =============================================================================

@tool
def analyze_project_patterns() -> str:
    """
    Analyze project patterns including dependencies, frameworks, and test setup.

    Returns comprehensive information about:
    - Installed packages (from virtual environment)
    - Project dependencies (from pyproject.toml, requirements.txt)
    - Detected frameworks (web, test, database, async)
    - Test patterns and conventions

    Use this to understand the project structure before generating code.

    Returns:
        JSON string with project analysis
    """
    output = ["=== PROJECT PATTERNS ===\n"]

    # Virtual Environment Packages
    if _state.venv_packages:
        output.append("INSTALLED PACKAGES (from virtual environment):")
        package_items = list(_state.venv_packages.items())[:10]
        for name, info in package_items:
            output.append(f"  • {name} v{info.version}")
            if info.test_utilities:
                test_utils = ', '.join(info.test_utilities)
                output.append(f"    Test utilities: {test_utils}")
            if info.decorators:
                decorators = ', '.join(info.decorators[:3])
                output.append(f"    Decorators: {decorators}")
        output.append("")

    # Dependencies (DependencyInfo dataclass - use attributes)
    if _state.dependencies:
        deps = _state.dependencies
        output.append("PROJECT DEPENDENCIES:")

        # Access dataclass attributes directly
        output.append(f"  Total: {len(deps.all_dependencies)}")

        if deps.web_framework:
            output.append(f"  Web framework: {deps.web_framework}")

        if deps.test_framework:
            output.append(f"  Test framework: {deps.test_framework}")

        if deps.database:
            output.append(f"  Database: {deps.database}")

        if deps.async_framework:
            output.append(f"  Async: {deps.async_framework}")

        output.append("")

    # Test Patterns (also likely a dataclass or dict - handle both)
    if _state.test_patterns:
        patterns = _state.test_patterns
        output.append("TEST PATTERNS:")

        # Check if it's a dataclass or dict
        if is_dataclass(patterns) and not isinstance(patterns, type):
            # It's a dataclass - use attributes
            output.append(f"  Framework: {patterns.framework}")    # type: ignore
            output.append(f"  Fixture style: {patterns.fixture_style}")    # type: ignore
            output.append(f"  Assertion style: {patterns.assertion_style}")    # type: ignore
            output.append(f"  Naming convention: {patterns.naming_convention}")    # type: ignore

            if patterns.uses_mocking:    # type: ignore
                output.append(f"  Mocking: {patterns.mocking_library}")    # type: ignore

            output.append(f"  Total test files: {patterns.total_test_files}")    # type: ignore
            output.append(f"  Total test functions: {patterns.total_test_functions}")    # type: ignore
        elif isinstance(patterns, dict):
            # It's a dict - use .get()
            output.append(f"  Framework: {patterns.get('framework', 'unknown')}")
            output.append(f"  Fixture style: {patterns.get('fixture_style', 'unknown')}")
            output.append(f"  Assertion style: {patterns.get('assertion_style', 'unknown')}")
            output.append(f"  Naming convention: {patterns.get('naming_convention', 'unknown')}")

            if patterns.get('uses_mocking'):
                output.append(f"  Mocking: {patterns.get('mocking_library', 'unknown')}")

            output.append(f"  Total test files: {patterns.get('total_test_files', 0)}")
            output.append(f"  Total test functions: {patterns.get('total_test_functions', 0)}")
        else:
            raise ValueError("Unknown data format!")

        output.append("")

    return "\n".join(output)


@tool
def get_codebase_statistics() -> str:
    """
    Get overall codebase statistics.

    Returns:
        Statistics about files, functions, classes, complexity, etc.
    """
    if not _state.codebase:
        return "Error: Codebase not loaded"

    cb = _state.codebase

    output = ["=== CODEBASE STATISTICS ===\n", f"Total files: {cb.total_files}",
              f"Total functions: {cb.total_functions}", f"Total classes: {cb.total_classes}",
              f"Total lines of code: {cb.total_lines}", ""]

    # Complexity distribution
    if cb.total_functions > 0:
        all_funcs = []
        for file_info in cb.files.values():
            all_funcs.extend(file_info.get_all_functions())

        complexities = [f.cyclomatic_complexity for f in all_funcs]
        avg_complexity = sum(complexities) / len(complexities)
        max_complexity = max(complexities)

        output.append(f"Average complexity: {avg_complexity:.2f}")
        output.append(f"Maximum complexity: {max_complexity}")

        # Complex functions (complexity > 10)
        complex_funcs = [f for f in all_funcs if f.cyclomatic_complexity > 10]
        if complex_funcs:
            output.append(f"\nComplex functions (>10): {len(complex_funcs)}")
            for func in complex_funcs[:5]:
                output.append(f"  • {func.name} (complexity: {func.cyclomatic_complexity})")

    return "\n".join(output)


# =============================================================================
# FUNCTION ANALYSIS TOOLS
# =============================================================================

@tool
def get_function_dependencies(file_path: str, function_name: str) -> str:
    """
    Get dependencies of a specific function (what it calls).

    Args:
        file_path: Relative path to file (e.g., "src/auth/login.py")
        function_name: Name of the function

    Returns:
        List of functions this function calls

    Examples:
        get_function_dependencies("src/auth.py", "login")
    """
    if not _state.codebase:
        return "Error: Codebase not loaded"

    # Find the function
    target_funcs = _state.codebase.find_function(function_name)

    if not target_funcs:
        return f"Function '{function_name}' not found"

    # Filter by file path if specified
    if file_path:
        target_funcs = [f for f in target_funcs if file_path in str(f.file_path)]

    if not target_funcs:
        return f"Function '{function_name}' not found in '{file_path}'"

    func = target_funcs[0]

    output = [f"Dependencies of {func.name}:", f"Location: {func.file_path}:{func.line_start}", ""]

    if func.calls:
        output.append(f"Calls {len(func.calls)} functions:")
        for call in func.calls:
            output.append(f"  • {call}")
    else:
        output.append("No function calls detected")

    return "\n".join(output)


@tool
def analyze_function_complexity(file_path: str, function_name: str) -> str:
    """
    Analyze complexity metrics for a specific function.

    Args:
        file_path: Relative path to file
        function_name: Name of the function

    Returns:
        Complexity metrics and recommendations

    Examples:
        analyze_function_complexity("src/utils.py", "process_data")
    """
    if not _state.codebase:
        return "Error: Codebase not loaded"

    # Find the function
    target_funcs = _state.codebase.find_function(function_name)

    if not target_funcs:
        return f"Function '{function_name}' not found"

    if file_path:
        target_funcs = [f for f in target_funcs if file_path in str(f.file_path)]

    if not target_funcs:
        return f"Function '{function_name}' not found in '{file_path}'"

    func = target_funcs[0]

    output = [f"Complexity Analysis: {func.name}", f"Location: {func.file_path}:{func.line_start}", "",
              f"Cyclomatic Complexity: {func.cyclomatic_complexity}",
              f"Cognitive Complexity: {func.cognitive_complexity}", f"Lines of Code: {func.lines_of_code}", ""]

    # Recommendations
    if func.cyclomatic_complexity > 10:
        output.append("⚠ HIGH COMPLEXITY - Consider refactoring")
    elif func.cyclomatic_complexity > 5:
        output.append("⚠ MODERATE COMPLEXITY - May benefit from simplification")
    else:
        output.append("✓ LOW COMPLEXITY - Easy to test")

    return "\n".join(output)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_all_tools() -> List:
    """
    Get all available tools for the agent.

    Returns:
        List of LangChain tools
    """
    return [
        search_codebase,
        find_package_usage_examples,
        analyze_project_patterns,
        get_codebase_statistics,
        get_function_dependencies,
        analyze_function_complexity,
    ]


def get_tool_descriptions() -> str:
    """
    Get human-readable descriptions of all tools.

    Returns:
        Formatted string describing all tools
    """
    tools = get_all_tools()

    output = ["=== AVAILABLE TOOLS ===\n"]

    for tool_func in tools:
        output.append(f"• {tool_func.name}")
        output.append(f"  {tool_func.description}")
        output.append("")

    return "\n".join(output)
