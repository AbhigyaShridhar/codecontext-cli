from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage
import operator


class CodebaseState(TypedDict):
    """
    Base state shared by all agent workflows

    Attributes:
        project_root: Absolute path to project root
        codebase: Parsed codebase representation (dict form of Codebase class)
        query: User's question or request
        query_type: What kind of task this is
        messages: Conversation history (accumulated)
        search_results: Results from code search
        analysis: Analysis results (flexible dict)
        final_output: Final result to show user
        confidence_score: Agent's confidence in result (0.0-1.0)
        needs_iteration: Whether agent should iterate again
    """
    # Project context
    project_root: str
    codebase: dict  # Will be populated with parsed codebase

    # Query information
    query: str
    query_type: Literal["qa", "test_gen", "git_analysis"]

    # Agent working memory
    # IMPORTANT: Annotated with operator.add means lists accumulate
    # Without this, each node would overwrite the messages list
    messages: Annotated[Sequence[BaseMessage], operator.add]

    search_results: list[dict]
    analysis: dict  # Flexible storage for agent-specific analysis

    # Output
    final_output: str
    confidence_score: float
    needs_iteration: bool


class QAState(CodebaseState):
    """
    State for Q&A agent workflow

    Extends CodebaseState with Q&A-specific fields.

    Flow:
    1. User asks question
    2. Agent analyzes query complexity
    3. Performs search (possibly multiple times)
    4. Synthesizes answer
    5. Validates answer quality

    Example state progression:
        Initial: {query: "How does auth work?", search_attempts: 0}
        After search: {search_results: [...], search_attempts: 1}
        After synthesis: {final_output: "Auth works by...", answer_quality: 0.85}
    """

    # Query analysis results
    needs_semantic_search: bool
    query_complexity: Literal["simple", "medium", "complex"]

    # Search tracking
    search_attempts: int
    max_search_attempts: int  # 3 standard retries

    # Quality tracking
    answer_quality: float  # 0.0 to 1.0


class TestGenState(CodebaseState):
    """
    State for test generation agent workflow

    Flow:
    1. Analyze function to test
    2. Gather context (dependencies, similar tests)
    3. Create test plan
    4. Generate test code
    5. Validate quality
    6. If quality < threshold, improve and retry

    Example state progression:
        Initial: {target_function: "login", generation_attempt: 0}
        After analysis: {function_info: {...}, dependencies: [...]}
        After generation: {test_code: "def test_login...", test_quality_score: 0.7}
        After improvement: {test_code: "def test_login...", test_quality_score: 0.9}
    """

    # Target information
    target_file: str
    target_function: str
    function_info: dict  # Contains: name, signature, code, complexity...

    # Context for generation
    dependencies: list[dict]
    similar_tests: list[str]  # Example tests for style reference and more context on workflow

    # Generation tracking
    generation_attempt: int
    test_code: str
    validation_result: dict

    # Quality metrics
    test_quality_score: float  # 0.0 to 1.0


class GitAnalysisState(CodebaseState):
    """
    State for git analysis agent workflow

    Flow:
    1. Analyze commit/diff
    2. Identify changed functions
    3. Calculate risk scores
    4. Suggest which functions need tests
    5. Optionally auto-generate tests

    Example state progression:
        Initial: {commit_hash: "abc123"}
        After analysis: {files_changed: [...], risk_score: 75.0}
        After suggestions: {test_suggestions: ["Test process_payment()"]}
        After generation: {tests_generated: ["test_payment.py"]}
    """

    # Git information
    commit_hash: str
    commit_analysis: dict  # Full CommitAnalysis object as dict

    # Change tracking
    files_changed: list[dict]
    functions_changed: list[dict]

    # Risk assessment
    risk_score: float  # 0.0 to 100.0
    test_suggestions: list[str]

    # Action tracking
    tests_generated: list[str]  # Paths to generated test files


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_qa_initial_state(
        project_root: str,
        codebase: dict,
        query: str,
        max_search_attempts: int = 3
) -> QAState:
    """
    Create initial state for Q&A agent

    Args:
        project_root: Absolute path to project
        codebase: Parsed codebase (as dict)
        query: User's question
        max_search_attempts: Max search iterations

    Returns:
        QAState ready for agent execution

    Example:
        >>> state = create_qa_initial_state(
        ...     project_root="/path/to/project",
        ...     codebase={...},
        ...     query="How does authentication work?"
        ... )
    """
    return QAState(
        # Base state
        project_root=project_root,
        codebase=codebase,
        query=query,
        query_type="qa",
        messages=[],
        search_results=[],
        analysis={},
        final_output="",
        confidence_score=0.0,
        needs_iteration=False,

        # Q&A specific
        needs_semantic_search=False,
        query_complexity="medium",
        search_attempts=0,
        max_search_attempts=max_search_attempts,
        answer_quality=0.0,
    )


def create_test_gen_initial_state(
        project_root: str,
        codebase: dict,
        target_file: str,
        target_function: str,
        function_info: dict
) -> TestGenState:
    """
    Create initial state for test generation agent

    Args:
        project_root: Absolute path to project
        codebase: Parsed codebase (as dict)
        target_file: File containing function to test
        target_function: Name of function to test
        function_info: Function metadata (name, code, complexity, etc.)

    Returns:
        TestGenState ready for agent execution

    Example:
        >>> state = create_test_gen_initial_state(
        ...     project_root="/path/to/project",
        ...     codebase={...},
        ...     target_file="auth.py",
        ...     target_function="login",
        ...     function_info={"name": "login", "code": "...", ...}
        ... )
    """
    return TestGenState(
        # Base state
        project_root=project_root,
        codebase=codebase,
        query="",  # Not used for test generation
        query_type="test_gen",
        messages=[],
        search_results=[],
        analysis={},
        final_output="",
        confidence_score=0.0,
        needs_iteration=False,

        # Test gen specific
        target_file=target_file,
        target_function=target_function,
        function_info=function_info,
        dependencies=[],
        similar_tests=[],
        generation_attempt=0,
        test_code="",
        validation_result={},
        test_quality_score=0.0,
    )


def create_git_analysis_initial_state(
        project_root: str,
        codebase: dict,
        commit_hash: str
) -> GitAnalysisState:
    """
    Create initial state for git analysis agent

    Args:
        project_root: Absolute path to project
        codebase: Parsed codebase (as dict)
        commit_hash: Git commit to analyze

    Returns:
        GitAnalysisState ready for agent execution
    """
    return GitAnalysisState(
        # Base state
        project_root=project_root,
        codebase=codebase,
        query="",  # Not used for git analysis
        query_type="git_analysis",
        messages=[],
        search_results=[],
        analysis={},
        final_output="",
        confidence_score=0.0,
        needs_iteration=False,

        # Git analysis specific
        commit_hash=commit_hash,
        commit_analysis={},
        files_changed=[],
        functions_changed=[],
        risk_score=0.0,
        test_suggestions=[],
        tests_generated=[],
    )
