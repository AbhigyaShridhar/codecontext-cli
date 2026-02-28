"""
Test pattern detection - learn from existing tests

Analyzes existing test files to extract:
- Test framework (pytest vs. unittest)
- Fixture patterns
- Assertion styles
- Mocking approaches
- Naming conventions
- Example test code

This enables the agent to generate tests that match the project's style.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field

from codecontext.core.codebase import Codebase, FileInfo


@dataclass
class TestPatterns:
    """
    Learned test patterns from existing tests

    Everything learned by analyzing actual test code,
    not hardcoded assumptions.
    """

    # Framework detection
    framework: str = "pytest"  # "pytest" or "unittest"

    # Fixture patterns
    uses_fixtures: bool = False
    fixture_style: str = "none"  # "pytest", "unittest-setup", "none"

    # Assertion style
    assertion_style: str = "assert"  # "assert" or "self.assert*"

    # Mocking
    uses_mocking: bool = False
    mocking_library: Optional[str] = None  # "unittest.mock", "pytest-mock", etc.

    # Naming conventions
    naming_convention: str = "test_*"  # "test_*" or "*_test"
    file_naming: str = "test_*.py"  # "test_*.py" or "*_test.py"

    # Advanced pytest features
    uses_parametrize: bool = False
    uses_markers: bool = False

    # Example test code (for reference)
    example_tests: List[str] = field(default_factory=list)

    # Statistics
    total_test_files: int = 0
    total_test_functions: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for caching"""
        return {
            "framework": self.framework,
            "uses_fixtures": self.uses_fixtures,
            "fixture_style": self.fixture_style,
            "assertion_style": self.assertion_style,
            "uses_mocking": self.uses_mocking,
            "mocking_library": self.mocking_library,
            "naming_convention": self.naming_convention,
            "file_naming": self.file_naming,
            "uses_parametrize": self.uses_parametrize,
            "uses_markers": self.uses_markers,
            "example_count": len(self.example_tests),
            "total_test_files": self.total_test_files,
            "total_test_functions": self.total_test_functions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TestPatterns":
        """Create from dictionary (without example_tests for size)"""
        return cls(
            framework=data.get("framework", "pytest"),
            uses_fixtures=data.get("uses_fixtures", False),
            fixture_style=data.get("fixture_style", "none"),
            assertion_style=data.get("assertion_style", "assert"),
            uses_mocking=data.get("uses_mocking", False),
            mocking_library=data.get("mocking_library"),
            naming_convention=data.get("naming_convention", "test_*"),
            file_naming=data.get("file_naming", "test_*.py"),
            uses_parametrize=data.get("uses_parametrize", False),
            uses_markers=data.get("uses_markers", False),
            example_tests=[],  # Don't cache examples (too large)
            total_test_files=data.get("total_test_files", 0),
            total_test_functions=data.get("total_test_functions", 0),
        )


def detect_test_patterns(codebase: Codebase) -> TestPatterns:
    """
    Detect test patterns from existing tests

    Analyzes all test files in the codebase to learn patterns.

    Args:
        codebase: Parsed codebase

    Returns:
        TestPatterns with learned conventions

    Example:
        >>> patterns = detect_test_patterns(codebase)
        >>> print(patterns.framework)
        'pytest'
        >>> print(patterns.fixture_style)
        'pytest'
        >>> print(patterns.example_tests[0])
        'def test_login(client): ...'
    """
    # Find test files
    test_files = find_test_files(codebase)

    if not test_files:
        # No tests exist - return defaults
        return _default_patterns()

    # Analyze test files
    framework = detect_framework(test_files)
    fixture_info = _analyze_fixtures(test_files)
    assertion_style = detect_assertion_style(test_files)
    mocking_info = _analyze_mocking(test_files)
    naming = detect_naming_convention(test_files)
    file_naming = _detect_file_naming(test_files)
    uses_param = _check_parametrize(test_files)
    uses_markers = _check_markers(test_files)
    examples = _extract_best_examples(test_files, count=5)

    # Count test functions
    total_funcs = sum(len(f.get_all_functions()) for f in test_files)

    return TestPatterns(
        framework=framework,
        uses_fixtures=fixture_info["uses_fixtures"],
        fixture_style=fixture_info["style"],
        assertion_style=assertion_style,
        uses_mocking=mocking_info["uses_mocking"],
        mocking_library=mocking_info["library"],
        naming_convention=naming,
        file_naming=file_naming,
        uses_parametrize=uses_param,
        uses_markers=uses_markers,
        example_tests=examples,
        total_test_files=len(test_files),
        total_test_functions=total_funcs,
    )


def find_test_files(codebase: Codebase) -> List[FileInfo]:
    """
    Find all test files in codebase

    Looks for common test file patterns:
    - test_*.py
    - *_test.py
    - Files in tests/ directory
    - Files in test/ directory
    """
    test_files = []

    for file_info in codebase.files.values():
        file_path = str(file_info.path).lower()
        file_name = file_info.path.name.lower()

        # Common test file patterns
        is_test_file = (
            # Naming patterns
            file_name.startswith("test_") or
            file_name.endswith("_test.py") or
            # Directory patterns
            "/test/" in file_path or
            "/tests/" in file_path or
            "\\test\\" in file_path or
            "\\tests\\" in file_path
        )

        if is_test_file:
            test_files.append(file_info)

    return test_files


def detect_framework(test_files: List[FileInfo]) -> str:
    """
    Detect if using pytest or unittest

    Looks at:
    - Imports (pytest vs unittest)
    - Decorators (@pytest.fixture vs setUp)
    - Base classes (TestCase)
    """
    pytest_indicators = 0
    unittest_indicators = 0

    for file in test_files:
        # Check imports
        imports_str = " ".join(file.imports)
        if "pytest" in imports_str:
            pytest_indicators += 2
        if "unittest" in imports_str:
            unittest_indicators += 2

        # Check for pytest fixtures
        for func in file.get_all_functions():
            if "@pytest.fixture" in " ".join(func.decorators):
                pytest_indicators += 1
            if "@pytest.mark" in " ".join(func.decorators):
                pytest_indicators += 1

        # Check for unittest.TestCase
        for cls in file.classes:
            bases_str = " ".join(cls.bases)
            if "TestCase" in bases_str or "unittest.TestCase" in bases_str:
                unittest_indicators += 2

    # Default to pytest if no clear indicator
    if pytest_indicators == 0 and unittest_indicators == 0:
        return "pytest"

    return "pytest" if pytest_indicators >= unittest_indicators else "unittest"


def _analyze_fixtures(test_files: List[FileInfo]) -> Dict:
    """
    Analyze fixture usage

    Returns:
        Dict with:
        - uses_fixtures: bool
        - style: "pytest", "unittest-setup", or "none"
    """
    uses_pytest_fixtures = False
    uses_unittest_setup = False

    for file in test_files:
        # Pytest fixtures
        for func in file.get_all_functions():
            if "@pytest.fixture" in " ".join(func.decorators):
                uses_pytest_fixtures = True

        # Unittest setUp/tearDown
        for cls in file.classes:
            method_names = [m.name for m in cls.methods]
            if any(name in method_names for name in ["setUp", "setUpClass", "tearDown", "tearDownClass"]):
                uses_unittest_setup = True

    if uses_pytest_fixtures:
        return {"uses_fixtures": True, "style": "pytest"}
    elif uses_unittest_setup:
        return {"uses_fixtures": True, "style": "unittest-setup"}
    else:
        return {"uses_fixtures": False, "style": "none"}


def detect_assertion_style(test_files: List[FileInfo]) -> str:
    """
    Detect assertion style

    Returns:
        "assert" (pytest style) or "self.assert*" (unittest style)
    """
    assert_count = 0
    self_assert_count = 0

    for file in test_files:
        for func in file.get_all_functions():
            # Count assert statements (pytest style)
            if "assert " in func.code:
                assert_count += func.code.count("assert ")

            # Count 'self.assert' statements (unittest style)
            if "self.assert" in func.code:
                self_assert_count += func.code.count("self.assert")

    return "assert" if assert_count >= self_assert_count else "self.assert*"


def _analyze_mocking(test_files: List[FileInfo]) -> Dict:
    """
    Analyze mocking usage

    Returns:
        Dict with:
        - uses_mocking: bool
        - library: "unittest.mock", "pytest-mock", or None
    """
    uses_mock = False
    library = None

    for file in test_files:
        imports_str = " ".join(file.imports)

        # unittest.mock
        if "unittest.mock" in imports_str or "from mock import" in imports_str:
            uses_mock = True
            library = "unittest.mock"

        # pytest-mock
        if "pytest_mock" in imports_str or "mocker" in imports_str:
            uses_mock = True
            library = "pytest-mock"

        # Check for mock/patch usage in code
        for func in file.get_all_functions():
            if any(keyword in func.code for keyword in ["@patch", "@mock", "mocker"]):
                uses_mock = True

    return {"uses_mocking": uses_mock, "library": library}


def detect_naming_convention(test_files: List[FileInfo]) -> str:
    """
    Detect test function naming convention

    Returns:
        "test_*" or "*_test"
    """
    test_prefix_count = 0
    test_suffix_count = 0

    for file in test_files:
        for func in file.get_all_functions():
            if func.name.startswith("test_"):
                test_prefix_count += 1
            elif func.name.endswith("_test"):
                test_suffix_count += 1

    return "test_*" if test_prefix_count >= test_suffix_count else "*_test"


def _detect_file_naming(test_files: List[FileInfo]) -> str:
    """
    Detect test file naming convention

    Returns:
        "test_*.py" or "*_test.py"
    """
    test_prefix_count = 0
    test_suffix_count = 0

    for file in test_files:
        file_name = file.path.name
        if file_name.startswith("test_"):
            test_prefix_count += 1
        elif file_name.endswith("_test.py"):
            test_suffix_count += 1

    return "test_*.py" if test_prefix_count >= test_suffix_count else "*_test.py"


def _check_parametrize(test_files: List[FileInfo]) -> bool:
    """Check if using pytest.mark.parametrize"""
    for file in test_files:
        for func in file.get_all_functions():
            if "@pytest.mark.parametrize" in " ".join(func.decorators):
                return True
    return False


def _check_markers(test_files: List[FileInfo]) -> bool:
    """Check if using pytest markers (skip, xfail, etc.)"""
    for file in test_files:
        for func in file.get_all_functions():
            decorators_str = " ".join(func.decorators)
            if any(marker in decorators_str for marker in ["@pytest.mark", "@skip", "@xfail"]):
                return True
    return False


def _extract_best_examples(test_files: List[FileInfo], count: int = 5) -> List[str]:
    """
    Extract best example test functions

    Prefers:
    - Medium complexity (not too simple, not too complex)
    - Well-documented (has docstring)
    - Proper test functions (name starts with test_)

    Args:
        test_files: List of test files
        count: Number of examples to extract

    Returns:
        List of test function code strings
    """
    examples = []

    for file in test_files:
        for func in file.get_all_functions():
            # Only include functions that look like tests
            if not func.name.startswith("test_"):
                continue

            # Skip very simple tests (complexity 1)
            if func.cyclomatic_complexity <= 1:
                continue

            # Skip very complex tests (complexity > 10)
            if func.cyclomatic_complexity > 10:
                continue

            # Prefer tests with docstrings
            if func.docstring:
                examples.append(func.code)
            # Also include good tests without docstrings
            elif 3 <= func.cyclomatic_complexity <= 5:
                examples.append(func.code)

            if len(examples) >= count:
                return examples

    # If we didn't find enough good examples, include any test
    if len(examples) < count:
        for file in test_files:
            for func in file.get_all_functions():
                if func.name.startswith("test_") and func.code not in examples:
                    examples.append(func.code)
                    if len(examples) >= count:
                        return examples

    return examples


def _default_patterns() -> TestPatterns:
    """
    Default patterns when no tests exist

    Uses modern best practices (pytest).
    """
    return TestPatterns(
        framework="pytest",
        uses_fixtures=True,
        fixture_style="pytest",
        assertion_style="assert",
        uses_mocking=False,
        mocking_library=None,
        naming_convention="test_*",
        file_naming="test_*.py",
        uses_parametrize=False,
        uses_markers=False,
        example_tests=[],
        total_test_files=0,
        total_test_functions=0,
    )
