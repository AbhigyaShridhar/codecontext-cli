# tests/unit/analysis/test_dependencies.py

"""
Tests for dependency parser
"""


from codecontext.analysis.dependencies import (
    parse_dependencies,
    extract_package_name,
    DependencyInfo,
)


class TestExtractPackageName:
    """Tests for package name extraction"""

    def test_simple_name(self):
        """Test simple package name"""
        assert extract_package_name("django") == "django"

    def test_version_specifier(self):
        """Test package with The version"""
        assert extract_package_name("django>=3.0") == "django"
        assert extract_package_name("requests==2.28.0") == "requests"
        assert extract_package_name("pytest>=7.0,<8.0") == "pytest"

    def test_extras(self):
        """Test package with extras"""
        assert extract_package_name("requests[security]") == "requests"
        assert extract_package_name("fastapi[all]") == "fastapi"

    def test_complex_specifier(self):
        """Test complex version specifier"""
        assert extract_package_name("numpy>=1.20.0,<2.0.0") == "numpy"
        assert extract_package_name("scikit-learn~=1.0") == "scikit-learn"

    def test_underscore_normalization(self):
        """Test underscore to hyphen normalization"""
        assert extract_package_name("scikit_learn") == "scikit-learn"


class TestDependencyInfo:
    """Tests for DependencyInfo dataclass"""

    def test_to_dict(self):
        """Test conversion to dictionary"""
        info = DependencyInfo()
        info.all_dependencies = {"django", "pytest"}
        info.dev_dependencies = {"pytest"}
        info.web_framework = "django"
        info.test_framework = "pytest"

        data = info.to_dict()

        assert set(data["all_dependencies"]) == {"django", "pytest"}
        assert set(data["dev_dependencies"]) == {"pytest"}
        assert data["web_framework"] == "django"
        assert data["test_framework"] == "pytest"

    def test_from_dict(self):
        """Test creation from dictionary"""
        data = {
            "all_dependencies": ["django", "pytest"],
            "dev_dependencies": ["pytest"],
            "web_framework": "django",
            "test_framework": "pytest",
            "database": "postgresql",
            "async_framework": None,
        }

        info = DependencyInfo.from_dict(data)

        assert info.all_dependencies == {"django", "pytest"}
        assert info.dev_dependencies == {"pytest"}
        assert info.web_framework == "django"
        assert info.test_framework == "pytest"


class TestParsePyprojectToml:
    """Tests for pyproject.toml parsing"""

    def test_pep621_format(self, tmp_path):
        """Test PEP 621 format"""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[project]
name = "my-project"
dependencies = [
    "django>=4.0",
    "requests>=2.28",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "black>=23.0",
]
""")

        info = parse_dependencies(tmp_path)

        assert "django" in info.all_dependencies
        assert "requests" in info.all_dependencies
        assert "pytest" in info.all_dependencies
        assert "black" in info.all_dependencies
        assert "pytest" in info.dev_dependencies
        assert "black" in info.dev_dependencies

    def test_poetry_format(self, tmp_path):
        """Test Poetry format"""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[tool.poetry]
name = "my-project"

[tool.poetry.dependencies]
python = "^3.10"
fastapi = "^0.100.0"
uvicorn = "^0.23.0"

[tool.poetry.dev-dependencies]
pytest = "^7.4.0"
mypy = "^1.5.0"
""")

        info = parse_dependencies(tmp_path)

        assert "fastapi" in info.all_dependencies
        assert "uvicorn" in info.all_dependencies
        assert "pytest" in info.all_dependencies
        assert "mypy" in info.all_dependencies
        assert "pytest" in info.dev_dependencies


class TestParseRequirementsTxt:
    """Tests for requirements.txt parsing"""

    def test_simple_requirements(self, tmp_path):
        """Test simple requirements.txt"""
        requirements = tmp_path / "requirements.txt"
        requirements.write_text("""
django==4.2.0
requests>=2.28.0
pytest
""")

        info = parse_dependencies(tmp_path)

        assert "django" in info.all_dependencies
        assert "requests" in info.all_dependencies
        assert "pytest" in info.all_dependencies

    def test_with_comments(self, tmp_path):
        """Test requirements with comments"""
        requirements = tmp_path / "requirements.txt"
        requirements.write_text("""
# Web framework
django==4.2.0

# HTTP client
requests>=2.28.0

# Testing
pytest
""")

        info = parse_dependencies(tmp_path)

        assert "django" in info.all_dependencies
        assert "requests" in info.all_dependencies
        assert "pytest" in info.all_dependencies

    def test_dev_requirements(self, tmp_path):
        """Test dev requirements"""
        # Main requirements
        requirements = tmp_path / "requirements.txt"
        requirements.write_text("django==4.2.0\n")

        # Dev requirements
        req_dev = tmp_path / "requirements-dev.txt"
        req_dev.write_text("pytest>=7.0\nblack>=23.0\n")

        info = parse_dependencies(tmp_path)

        assert "django" in info.all_dependencies
        assert "pytest" in info.all_dependencies
        assert "black" in info.all_dependencies
        assert "pytest" in info.dev_dependencies
        assert "black" in info.dev_dependencies
        assert "django" not in info.dev_dependencies


class TestFrameworkDetection:
    """Tests for framework auto-detection"""

    def test_detect_django(self, tmp_path):
        """Test Django detection"""
        requirements = tmp_path / "requirements.txt"
        requirements.write_text("django==4.2.0\npsycopg2>=2.9.0\n")

        info = parse_dependencies(tmp_path)

        assert info.web_framework == "django"
        assert info.database == "django-orm"  # Django has its own ORM

    def test_detect_fastapi(self, tmp_path):
        """Test FastAPI detection"""
        requirements = tmp_path / "requirements.txt"
        requirements.write_text("fastapi==0.100.0\npytest>=7.0\n")

        info = parse_dependencies(tmp_path)

        assert info.web_framework == "fastapi"
        assert info.test_framework == "pytest"
        assert info.async_framework == "asyncio"  # FastAPI uses asyncio

    def test_detect_flask(self, tmp_path):
        """Test Flask detection"""
        requirements = tmp_path / "requirements.txt"
        requirements.write_text("flask==2.3.0\nsqlalchemy>=2.0.0\n")

        info = parse_dependencies(tmp_path)

        assert info.web_framework == "flask"
        assert info.database == "sqlalchemy"

    def test_detect_pytest(self, tmp_path):
        """Test pytest detection"""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[project]
name = "test"

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-cov>=4.0"]
""")

        info = parse_dependencies(tmp_path)

        assert info.test_framework == "pytest"

    def test_priority_order(self, tmp_path):
        """Test framework detection priority"""
        # Django has priority over FastAPI
        requirements = tmp_path / "requirements.txt"
        requirements.write_text("django==4.2.0\nfastapi==0.100.0\n")

        info = parse_dependencies(tmp_path)

        assert info.web_framework == "django"  # Django has higher priority


class TestParseDependencies:
    """Integration tests for parse_dependencies"""

    def test_empty_directory(self, tmp_path):
        """Test empty directory (no dependency files)"""
        info = parse_dependencies(tmp_path)

        assert len(info.all_dependencies) == 0
        assert info.web_framework is None
        assert info.test_framework is None

    def test_multiple_files(self, tmp_path):
        """Test multiple dependency files"""
        # Create pyproject.toml
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[project]
dependencies = ["fastapi>=0.100"]
""")

        # Create requirements.txt (should merge)
        requirements = tmp_path / "requirements.txt"
        requirements.write_text("pytest>=7.0\n")

        info = parse_dependencies(tmp_path)

        assert "fastapi" in info.all_dependencies
        assert "pytest" in info.all_dependencies
