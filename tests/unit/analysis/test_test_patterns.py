# tests/unit/analysis/test_test_patterns.py

"""
Tests for test pattern detector
"""

from codecontext.core.parser import CodebaseParser
from codecontext.analysis.test_patterns import (
    detect_test_patterns,
    find_test_files,
    detect_framework,
    detect_assertion_style,
    detect_naming_convention,
    TestPatterns,
)


class TestTestPatterns:
    """Tests for TestPatterns dataclass"""

    def test_to_dict(self):
        """Test conversion to dictionary"""
        patterns = TestPatterns(
            framework="pytest",
            uses_fixtures=True,
            fixture_style="pytest",
            assertion_style="assert",
            uses_mocking=True,
            mocking_library="pytest-mock",
            total_test_files=5,
            total_test_functions=20,
        )

        data = patterns.to_dict()

        assert data["framework"] == "pytest"
        assert data["uses_fixtures"] is True
        assert data["fixture_style"] == "pytest"
        assert data["total_test_files"] == 5
        assert data["total_test_functions"] == 20

    def test_from_dict(self):
        """Test creation from dictionary"""
        data = {
            "framework": "unittest",
            "uses_fixtures": True,
            "fixture_style": "unittest-setup",
            "assertion_style": "self.assert*",
            "uses_mocking": False,
            "mocking_library": None,
            "total_test_files": 3,
            "total_test_functions": 10,
        }

        patterns = TestPatterns.from_dict(data)

        assert patterns.framework == "unittest"
        assert patterns.uses_fixtures is True
        assert patterns.fixture_style == "unittest-setup"


class TestFindTestFiles:
    """Tests for finding test files"""

    def test_find_test_prefix(self, tmp_path):
        """Test finding test_*.py files"""
        # Create test file
        test_file = tmp_path / "test_login.py"
        test_file.write_text("def test_something(): pass")

        # Parse
        parser = CodebaseParser(tmp_path, ignore_patterns=[])
        codebase = parser.parse()

        # Find test files
        test_files = find_test_files(codebase)

        assert len(test_files) >= 1
        assert any("test_login.py" in str(f.path) for f in test_files)

    def test_find_test_suffix(self, tmp_path):
        """Test finding *_test.py files"""
        test_file = tmp_path / "login_test.py"
        test_file.write_text("def test_something(): pass")

        parser = CodebaseParser(tmp_path, ignore_patterns=[])
        codebase = parser.parse()

        test_files = find_test_files(codebase)

        assert len(test_files) >= 1
        assert any("login_test.py" in str(f.path) for f in test_files)

    def test_find_in_tests_directory(self, tmp_path):
        """Test finding files in tests/ directory"""
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        test_file = tests_dir / "test_auth.py"
        test_file.write_text("def test_something(): pass")

        parser = CodebaseParser(tmp_path, ignore_patterns=[])
        codebase = parser.parse()

        test_files = find_test_files(codebase)

        assert len(test_files) >= 1


class TestDetectFramework:
    """Tests for framework detection"""

    def test_detect_pytest_from_imports(self, tmp_path):
        """Test pytest detection from imports"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text("""
import pytest

def test_something():
    assert True
""")

        parser = CodebaseParser(tmp_path, ignore_patterns=[])
        codebase = parser.parse()
        test_files = find_test_files(codebase)

        framework = detect_framework(test_files)

        assert framework == "pytest"

    def test_detect_pytest_from_fixtures(self, tmp_path):
        """Test pytest detection from fixtures"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text("""
import pytest

@pytest.fixture
def client():
    return TestClient()

def test_something(client):
    assert client is not None
""")

        parser = CodebaseParser(tmp_path, ignore_patterns=[])
        codebase = parser.parse()
        test_files = find_test_files(codebase)

        framework = detect_framework(test_files)

        assert framework == "pytest"

    def test_detect_unittest_from_testcase(self, tmp_path):
        """Test unittest detection from TestCase"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text("""
import unittest

class TestLogin(unittest.TestCase):
    def test_something(self):
        self.assertTrue(True)
""")

        parser = CodebaseParser(tmp_path, ignore_patterns=[])
        codebase = parser.parse()
        test_files = find_test_files(codebase)

        framework = detect_framework(test_files)

        assert framework == "unittest"


class TestDetectAssertionStyle:
    """Tests for assertion style detection"""

    def test_detect_assert_style(self, tmp_path):
        """Test pytest assert style"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text("""
def test_something():
    result = 1 + 1
    assert result == 2
    assert result > 0
""")

        parser = CodebaseParser(tmp_path, ignore_patterns=[])
        codebase = parser.parse()
        test_files = find_test_files(codebase)

        style = detect_assertion_style(test_files)

        assert style == "assert"

    def test_detect_self_assert_style(self, tmp_path):
        """Test unittest assert style"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text("""
import unittest

class TestSomething(unittest.TestCase):
    def test_it(self):
        result = 1 + 1
        self.assertEqual(result, 2)
        self.assertTrue(result > 0)
""")

        parser = CodebaseParser(tmp_path, ignore_patterns=[])
        codebase = parser.parse()
        test_files = find_test_files(codebase)

        style = detect_assertion_style(test_files)

        assert style == "self.assert*"


class TestDetectNamingConvention:
    """Tests for naming convention detection"""

    def test_detect_test_prefix(self, tmp_path):
        """Test test_* convention"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text("""
def test_login():
    pass

def test_logout():
    pass

def test_signup():
    pass
""")

        parser = CodebaseParser(tmp_path, ignore_patterns=[])
        codebase = parser.parse()
        test_files = find_test_files(codebase)

        convention = detect_naming_convention(test_files)

        assert convention == "test_*"

    def test_detect_test_suffix(self, tmp_path):
        """Test *_test convention"""
        test_file = tmp_path / "example_test.py"
        test_file.write_text("""
def login_test():
    pass

def logout_test():
    pass

def signup_test():
    pass
""")

        parser = CodebaseParser(tmp_path, ignore_patterns=[])
        codebase = parser.parse()
        test_files = find_test_files(codebase)

        convention = detect_naming_convention(test_files)

        assert convention == "*_test"


class TestDetectTestPatterns:
    """Integration tests for detect_test_patterns"""

    def test_detect_pytest_patterns(self, tmp_path):
        """Test detecting pytest patterns"""
        # Create pytest test file
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        test_file = tests_dir / "test_auth.py"
        test_file.write_text("""
import pytest

@pytest.fixture
def user():
    return {"name": "John", "id": 1}

def test_user_creation(user):
    '''Test user creation'''
    assert user["name"] == "John"
    assert user["id"] == 1

@pytest.mark.parametrize("value,expected", [(1, 2), (2, 3)])
def test_increment(value, expected):
    '''Test increment'''
    assert value + 1 == expected
""")

        parser = CodebaseParser(tmp_path, ignore_patterns=[])
        codebase = parser.parse()

        patterns = detect_test_patterns(codebase)

        assert patterns.framework == "pytest"
        assert patterns.uses_fixtures is True
        assert patterns.fixture_style == "pytest"
        assert patterns.assertion_style == "assert"
        assert patterns.uses_parametrize is True
        assert patterns.naming_convention == "test_*"
        assert patterns.total_test_files >= 1
        assert patterns.total_test_functions >= 2

    def test_detect_unittest_patterns(self, tmp_path):
        """Test detecting unittest patterns"""
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        test_file = tests_dir / "test_auth.py"
        test_file.write_text("""
import unittest

class TestAuth(unittest.TestCase):
    def setUp(self):
        self.user = {"name": "John", "id": 1}

    def test_user_creation(self):
        '''Test user creation'''
        self.assertEqual(self.user["name"], "John")
        self.assertEqual(self.user["id"], 1)

    def test_user_name(self):
        '''Test user name'''
        self.assertIsNotNone(self.user["name"])
""")

        parser = CodebaseParser(tmp_path, ignore_patterns=[])
        codebase = parser.parse()

        patterns = detect_test_patterns(codebase)

        assert patterns.framework == "unittest"
        assert patterns.uses_fixtures is True
        assert patterns.fixture_style == "unittest-setup"
        assert patterns.assertion_style == "self.assert*"
        assert patterns.total_test_files >= 1

    def test_no_tests_returns_defaults(self, tmp_path):
        """Test default patterns when no tests exist"""
        # Create non-test file
        src_file = tmp_path / "main.py"
        src_file.write_text("def hello(): pass")

        parser = CodebaseParser(tmp_path, ignore_patterns=[])
        codebase = parser.parse()

        patterns = detect_test_patterns(codebase)

        # Should return sensible defaults
        assert patterns.framework == "pytest"
        assert patterns.fixture_style == "pytest"
        assert patterns.total_test_files == 0
        assert patterns.total_test_functions == 0
