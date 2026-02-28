"""
Dependency analysis - parse project dependencies

Supports:
- pyproject.toml (modern, PEP 621)
- requirements.txt (traditional)
- setup.py (legacy)
- Pipfile (pipenv)

Automatically detects:
- Web frameworks (Django, Flask, FastAPI)
- Test frameworks (pytest, unittest)
- Databases (SQLAlchemy, MongoDB, PostgreSQL)
- Async frameworks
"""

from pathlib import Path
from typing import Optional, Set
import re
from dataclasses import dataclass, field

# Try to import tomli for Python < 3.11, fallback to tomllib for 3.11+
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        tomllib = None  # type: ignore


@dataclass
class DependencyInfo:
    """
    Parsed dependency information

    Contains all dependencies and auto-detected frameworks.
    """
    all_dependencies: Set[str] = field(default_factory=set)
    dev_dependencies: Set[str] = field(default_factory=set)

    # Auto-detected frameworks
    web_framework: Optional[str] = None
    test_framework: Optional[str] = None
    database: Optional[str] = None
    async_framework: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "all_dependencies": list(self.all_dependencies),
            "dev_dependencies": list(self.dev_dependencies),
            "web_framework": self.web_framework,
            "test_framework": self.test_framework,
            "database": self.database,
            "async_framework": self.async_framework,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DependencyInfo":
        """Create from dictionary"""
        return cls(
            all_dependencies=set(data.get("all_dependencies", [])),
            dev_dependencies=set(data.get("dev_dependencies", [])),
            web_framework=data.get("web_framework"),
            test_framework=data.get("test_framework"),
            database=data.get("database"),
            async_framework=data.get("async_framework"),
        )


def parse_dependencies(project_root: Path) -> DependencyInfo:
    """
    Parse dependencies from all common formats

    Priority order:
    1. Pyproject.toml (modern standard)
    2. Requirements.txt (common)
    3. Setup.py (legacy)
    4. Pipfile (pipenv)

    Args:
        project_root: Project root directory

    Returns:
        DependencyInfo with parsed dependencies and detected frameworks

    Example:
        >>> example_info = parse_dependencies(Path("/path/to/project"))
        >>> print(example_info.web_framework)
        'fastapi'
        >>> print(example_info.test_framework)
        'pytest'
    """
    info = DependencyInfo()

    # Try pyproject.toml first (modern)
    pyproject = project_root / "pyproject.toml"
    if pyproject.exists():
        _parse_pyproject_toml(pyproject, info)

    # Try requirements.txt
    requirements = project_root / "requirements.txt"
    if requirements.exists():
        _parse_requirements_txt(requirements, info)

    # Try requirements-dev.txt
    req_dev = project_root / "requirements-dev.txt"
    if req_dev.exists():
        _parse_requirements_txt(req_dev, info, dev=True)

    # Try setup.py
    setup_py = project_root / "setup.py"
    if setup_py.exists():
        _parse_setup_py(setup_py, info)

    # Try Pipfile
    pipfile = project_root / "Pipfile"
    if pipfile.exists():
        _parse_pipfile(pipfile, info)

    # Analyze dependencies to detect frameworks
    _detect_frameworks(info)

    return info


def _parse_pyproject_toml(path: Path, info: DependencyInfo) -> None:
    """
    Parse pyproject.toml (PEP 621 format)

    Args:
        path: Path to pyproject.toml
        info: DependencyInfo to populate
    """
    if tomllib is None:
        return

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)

        # PEP 621 format
        if "project" in data:
            # Main dependencies
            deps = data["project"].get("dependencies", [])
            for dep in deps:
                pkg_name = extract_package_name(dep)
                info.all_dependencies.add(pkg_name)

            # Optional dependencies (dev, test, etc.)
            optional = data["project"].get("optional-dependencies", {})
            for group_name, group_deps in optional.items():
                for dep in group_deps:
                    pkg_name = extract_package_name(dep)
                    info.all_dependencies.add(pkg_name)
                    if group_name in ["dev", "test", "tests", "testing"]:
                        info.dev_dependencies.add(pkg_name)

        # Poetry format
        if "tool" in data and "poetry" in data["tool"]:
            poetry = data["tool"]["poetry"]

            # Main dependencies
            if "dependencies" in poetry:
                for dep_name in poetry["dependencies"].keys():
                    if dep_name != "python":  # Skip Python version
                        info.all_dependencies.add(dep_name.lower())

            # Dev dependencies
            if "dev-dependencies" in poetry:
                for dep_name in poetry["dev-dependencies"].keys():
                    pkg_name = dep_name.lower()
                    info.all_dependencies.add(pkg_name)
                    info.dev_dependencies.add(pkg_name)

    except (ValueError, TypeError, AttributeError, Exception):
        # Not a valid pyproject.toml or parsing failed
        pass


def _parse_requirements_txt(path: Path, info: DependencyInfo, dev: bool = False) -> None:
    """
    Parse requirements.txt format

    Args:
        path: Path to requirements file
        info: DependencyInfo to populate
        dev: Whether these are dev dependencies
    """
    try:
        content = path.read_text()
        for line in content.splitlines():
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            # Skip -e editable installs (just get package name)
            if line.startswith("-e"):
                # Extract package name from path/url
                # Example: -e git+https://github.com/user/repo.git#egg=package-name
                if "#egg=" in line:
                    pkg_name = line.split("#egg=")[-1]
                    info.all_dependencies.add(extract_package_name(pkg_name))
                continue

            # Skip other pip options
            if line.startswith("-"):
                continue

            pkg_name = extract_package_name(line)
            info.all_dependencies.add(pkg_name)
            if dev:
                info.dev_dependencies.add(pkg_name)

    except (ValueError, TypeError, AttributeError, Exception):
        pass


def _parse_setup_py(path: Path, info: DependencyInfo) -> None:
    """
    Parse setup.py (basic extraction using regex)

    Args:
        path: Path to the setup.py
        info: DependencyInfo to populate
    """
    try:
        content = path.read_text()

        # Find install_requires (basic regex)
        # Matches: install_requires=['package1', 'package2']
        install_requires_match = re.search(
            r'install_requires\s*=\s*\[(.*?)]',
            content,
            re.DOTALL
        )

        if install_requires_match:
            deps_str = install_requires_match.group(1)
            # Extract quoted strings
            deps = re.findall(r'["\']([^"\']+)["\']', deps_str)
            for dep in deps:
                pkg_name = extract_package_name(dep)
                info.all_dependencies.add(pkg_name)

        # Find extras_require for dev dependencies
        extras_match = re.search(
            r'extras_require\s*=\s*\{(.*?)}',
            content,
            re.DOTALL
        )

        if extras_match:
            extras_str = extras_match.group(1)
            # Look for dev/test groups
            dev_match = re.search(
                r'["\'](?:dev|test|testing)["\']:\s*\[(.*?)]',
                extras_str,
                re.DOTALL
            )
            if dev_match:
                deps_str = dev_match.group(1)
                deps = re.findall(r'["\']([^"\']+)["\']', deps_str)
                for dep in deps:
                    pkg_name = extract_package_name(dep)
                    info.all_dependencies.add(pkg_name)
                    info.dev_dependencies.add(pkg_name)

    except (ValueError, TypeError, AttributeError, Exception):
        pass


def _parse_pipfile(path: Path, info: DependencyInfo) -> None:
    """
    Parse Pipfile (pipenv format)

    Args:
        path: Path to Pipfile
        info: DependencyInfo to populate
    """
    if tomllib is None:
        return

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)

        # Main dependencies
        packages = data.get("packages", {})
        for pkg_name in packages.keys():
            info.all_dependencies.add(pkg_name.lower())

        # Dev dependencies
        dev_packages = data.get("dev-packages", {})
        for pkg_name in dev_packages.keys():
            pkg_name_lower = pkg_name.lower()
            info.all_dependencies.add(pkg_name_lower)
            info.dev_dependencies.add(pkg_name_lower)

    except (ValueError, TypeError, AttributeError, Exception):
        pass


def extract_package_name(dep_string: str) -> str:
    """
    Extract package name from dependency string

    Handles various formats:
    - "django>=3.0" → "django"
    - "requests[security]" → "requests"
    - "pytest>=7.0,<8.0" → "pytest"
    - "git+https://github.com/user/repo.git" → "repo"

    Args:
        dep_string: Dependency specification string

    Returns:
        Normalized package name (lowercase)
    """
    # Handle git URLs
    if dep_string.startswith("git+"):
        # Extract repo name from URL
        match = re.search(r'/([^/]+?)(?:\.git)?$', dep_string)
        if match:
            return match.group(1).lower()
        return dep_string.lower()

    # Remove version specifiers and extras
    # Split on common separators: >=, <=, ==, !=, ~=, [, <, >
    name = re.split(r'[><=!~\[]', dep_string)[0].strip()

    # Remove whitespace
    name = name.strip()

    # Lowercase and replace underscores with hyphens (normalize)
    return name.lower().replace('_', '-')


def _detect_frameworks(info: DependencyInfo) -> None:
    """
    Detect frameworks from the dependency list

    Uses common package names to identify:
    - Web frameworks
    - Test frameworks
    - Databases
    - Async frameworks

    Args:
        info: DependencyInfo to populate with detected frameworks
    """
    deps = info.all_dependencies

    # Web frameworks (priority order)
    if "django" in deps:
        info.web_framework = "django"
    elif "fastapi" in deps:
        info.web_framework = "fastapi"
    elif "flask" in deps:
        info.web_framework = "flask"
    elif "starlette" in deps:
        info.web_framework = "starlette"
    elif "tornado" in deps:
        info.web_framework = "tornado"
    elif "pyramid" in deps:
        info.web_framework = "pyramid"
    elif "bottle" in deps:
        info.web_framework = "bottle"

    # Test frameworks
    if "pytest" in deps:
        info.test_framework = "pytest"
    elif "unittest2" in deps:
        info.test_framework = "unittest"
    # Note: unittest is built-in, won't appear in deps

    # Databases (priority order)
    if "sqlalchemy" in deps:
        info.database = "sqlalchemy"
    elif "django" in deps:  # Django ORM
        info.database = "django-orm"
    elif "pymongo" in deps or "motor" in deps:
        info.database = "mongodb"
    elif "psycopg2" in deps or "psycopg" in deps or "psycopg2-binary" in deps:
        info.database = "postgresql"
    elif "mysql-connector-python" in deps or "pymysql" in deps:
        info.database = "mysql"
    elif "redis" in deps or "redis-py" in deps:
        info.database = "redis"

    # Async frameworks
    if "asyncio" in deps or "aiohttp" in deps or "fastapi" in deps:
        info.async_framework = "asyncio"
    elif "trio" in deps:
        info.async_framework = "trio"
    elif "tornado" in deps:
        info.async_framework = "tornado"
