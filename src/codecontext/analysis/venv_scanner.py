import ast
import sys
from pathlib import Path
from typing import Dict, List, Optional, cast
from dataclasses import dataclass, field
import importlib.metadata


@dataclass
class PackageInfo:
    """
    Information about an installed package

    Discovered by reading the actual package source code.
    """
    name: str
    version: str
    location: Path

    # Discovered patterns (from a reading source)
    test_utilities: List[str] = field(default_factory=list)  # e.g., ["TestClient", "fixture"]
    decorators: List[str] = field(default_factory=list)  # e.g., ["@app.route", "@pytest.fixture"]
    base_classes: List[str] = field(default_factory=list)  # e.g., ["TestCase", "BaseModel"]
    async_support: bool = False

    # Key modules/classes/functions
    main_classes: List[str] = field(default_factory=list)
    main_functions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for caching"""
        return {
            "name": self.name,
            "version": self.version,
            "location": str(self.location),
            "test_utilities": self.test_utilities,
            "decorators": self.decorators,
            "base_classes": self.base_classes,
            "async_support": self.async_support,
            "main_classes": self.main_classes,
            "main_functions": self.main_functions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PackageInfo":
        """Create from dictionary (from cache)"""
        return cls(
            name=data["name"],
            version=data["version"],
            location=Path(data["location"]),
            test_utilities=data.get("test_utilities", []),
            decorators=data.get("decorators", []),
            base_classes=data.get("base_classes", []),
            async_support=data.get("async_support", False),
            main_classes=data.get("main_classes", []),
            main_functions=data.get("main_functions", []),
        )


class VirtualEnvScanner:
    """
    Scans virtual environment to discover package patterns

    This is the KEY innovation - instead of hardcoding patterns,
    we READ the actual installed packages to learn their APIs.

    Example:
        >>> scanner = VirtualEnvScanner()
        >>> packages = scanner.scan()
        >>> fastapi = packages.get('fastapi')
        >>> if fastapi:
        ...     print(fastapi.test_utilities)
        ...     # ['TestClient']
        >>> print(fastapi.decorators)
        ['app.get', 'app.post', 'app.route']
    """

    def __init__(self, venv_path: Optional[Path] = None):
        """
        Initialize scanner

        Args:
            venv_path: Path to virtual environment
                      If None, detects current environment
        """
        self.venv_path = venv_path or self._detect_venv()
        self.site_packages = self._find_site_packages()

    @staticmethod
    def _detect_venv() -> Optional[Path]:
        """
        Detect current virtual environment

        Checks:
        1. sys.prefix (current Python)
        2. VIRTUAL_ENV environment variable
        3. Common locations (venv/, .venv/, env/)
        """
        # Check if we're in a venv
        if hasattr(sys, 'real_prefix') or (
                hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        ):
            return Path(sys.prefix)

        # Check VIRTUAL_ENV
        import os
        if 'VIRTUAL_ENV' in os.environ:
            return Path(os.environ['VIRTUAL_ENV'])

        # Check common locations relative to the current directory
        cwd = Path.cwd()
        for venv_name in ['venv', '.venv', 'env', '.env']:
            venv_path = cwd / venv_name
            if venv_path.exists():
                # Verify it's a venv by checking for Python executable
                python_paths = [
                    venv_path / 'bin' / 'python',
                    venv_path / 'Scripts' / 'python.exe',  # Windows
                ]
                if any(p.exists() for p in python_paths):
                    return venv_path

        return None

    def _find_site_packages(self) -> Optional[Path]:
        """
        Find site-packages directory in venv

        Returns path where packages are installed.
        """
        if not self.venv_path:
            # Use current Python's site-packages
            import site
            site_packages_list = site.getsitepackages()
            if site_packages_list:
                return Path(site_packages_list[0])
            return None

        # Look in venv
        lib_dir = self.venv_path / 'lib'

        # Windows uses different structure
        if not lib_dir.exists():
            lib_dir = self.venv_path / 'Lib'

        if not lib_dir.exists():
            return None

        # Find pythonX.Y directory
        for python_dir in lib_dir.iterdir():
            if python_dir.name.startswith('python'):
                site_packages = python_dir / 'site-packages'
                if site_packages.exists():
                    return site_packages

        # Windows directly has site-packages
        site_packages = lib_dir / 'site-packages'
        if site_packages.exists():
            return site_packages

        return None

    def scan(self, limit: Optional[int] = None) -> Dict[str, PackageInfo]:
        """
        Scan all installed packages

        Args:
            limit: Optional limit on number of packages to scan (for testing)

        Returns:
            Dict mapping package name to PackageInfo

        This is the main entry point. Call once and cache results.
        """
        if not self.site_packages:
            return {}

        packages = {}

        # Get a list of installed packages using importlib.metadata
        installed = self._get_installed_packages()

        count = 0
        for pkg_name, pkg_version, pkg_location in installed:
            if limit and count >= limit:
                break

            # Scan package to discover patterns
            info = self._scan_package(pkg_name, pkg_version, pkg_location)

            if info:
                packages[pkg_name] = info
                count += 1

        return packages

    def _get_installed_packages(self) -> List[tuple]:
        """
        Get a list of installed packages

        Returns:
            List of (name, version, location) tuples
        """
        installed = []

        try:
            # Use importlib.metadata to get installed packages
            for dist in importlib.metadata.distributions():
                name = dist.metadata['Name']
                version = dist.version

                # Get package location
                if dist.files:
                    # Get the first file's parent to determine package location
                    first_file = list(dist.files)[0]
                    location = Path(dist.locate_file(first_file)).parent

                    # Find the actual package directory
                    while location.name in ['site-packages', 'dist-info']:
                        location = location.parent

                    pkg_dir = self.site_packages / name.lower().replace('-', '_')
                    if pkg_dir.exists() and pkg_dir.is_dir():
                        installed.append((name.lower(), version, pkg_dir))

        except (ValueError, TypeError, AttributeError, Exception):
            # Fallback: scan site-packages directory directly
            if self.site_packages:
                for item in self.site_packages.iterdir():
                    if item.is_dir() and not item.name.startswith('_'):
                        pkg_name = item.name
                        version = self._extract_version(item / '__init__.py')
                        installed.append((pkg_name, version, item))

        return installed

    @staticmethod
    def _extract_version(init_file: Path) -> str:
        """Extract the version from __init__.py"""
        if not init_file.exists():
            return "unknown"

        try:
            content = init_file.read_text()

            # Look for __version__ = "x.y.z"
            for line in content.splitlines():
                if '__version__' in line and '=' in line:
                    version = line.split('=')[1].strip().strip('"\'')
                    return version

        except (ValueError, TypeError, AttributeError, Exception):
            pass

        return "unknown"

    def _scan_package(
            self,
            name: str,
            version: str,
            location: Path
    ) -> Optional[PackageInfo]:
        """
        Scan a package to discover patterns

        This is where the magic happens!
        We READ the package source to understand its API.

        Args:
            name: Package name
            version: Package version
            location: Path to package directory

        Returns:
            PackageInfo with discovered patterns, or None if scan fails
        """
        try:
            # Find the main module
            main_module = location / '__init__.py'

            if not main_module.exists():
                return None

            # Parse the module
            source = main_module.read_text(encoding='utf-8', errors='ignore')
            tree = ast.parse(source)

            # Extract patterns
            decorators = self._extract_decorators(tree)
            test_utilities = self._extract_test_utilities(tree, location)
            base_classes = self._extract_base_classes(tree)
            main_classes = self._extract_main_classes(tree)
            main_functions = self._extract_main_functions(tree)
            async_support = self._check_async_support(tree)

            return PackageInfo(
                name=name,
                version=version,
                location=location,
                test_utilities=test_utilities,
                decorators=decorators,
                base_classes=base_classes,
                async_support=async_support,
                main_classes=main_classes,
                main_functions=main_functions,
            )

        except (ValueError, TypeError, AttributeError, Exception):
            # Package couldn't be scanned, skip it
            # In production, you might want to log this
            return None

    def _extract_decorators(self, tree: ast.AST) -> List[str]:
        """
        Extract decorator patterns from package

        Looks for common decorator patterns:
        - @app.route (Flask)
        - @app.get (FastAPI)
        - @pytest.fixture
        - @pytest.mark
        """
        decorators = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    # Extract decorator name
                    dec_name = self._decorator_name(decorator)
                    if dec_name:
                        decorators.add(dec_name)

        return list(decorators)[:20]  # Limit to 20, common practise

    def _decorator_name(self, decorator: ast.expr) -> Optional[str]:
        """Extract decorator name from AST node"""
        try:
            if isinstance(decorator, ast.Name):
                return decorator.id
            elif isinstance(decorator, ast.Attribute):
                # e.g., pytest.fixture -> "pytest.fixture"
                return self._full_attribute_name(decorator)
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Attribute):
                    return self._full_attribute_name(decorator.func)
                elif isinstance(decorator.func, ast.Name):
                    return decorator.func.id
        except (ValueError, TypeError, AttributeError, Exception):
            pass
        return None

    @staticmethod
    def _full_attribute_name(node: ast.Attribute) -> str:
        """Get full dotted name from Attribute node"""
        parts = []
        current = node

        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value

        if isinstance(current, ast.Name):
            parts.append(current.id)

        return '.'.join(reversed(parts))

    @staticmethod
    def _extract_test_utilities(
            tree: ast.AST,
            pkg_location: Path
    ) -> List[str]:
        """
        Extract test utilities from package

        Looks for:
        - TestClient classes
        - Fixture functions
        - Test helpers
        """
        utilities = set()

        # Check for common test utility patterns
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # TestClient, AsyncTestClient, etc.
                name_lower = node.name.lower()
                if 'test' in name_lower and 'client' in name_lower:
                    utilities.add(node.name)
                elif node.name.endswith('TestCase'):
                    utilities.add(node.name)

            elif isinstance(node, ast.FunctionDef):
                # fixture, mark, etc.
                if node.name in ['fixture', 'mark', 'parametrize']:
                    utilities.add(node.name)

        # Check for testclient module (FastAPI pattern)
        testclient_module = pkg_location / 'testclient.py'
        if testclient_module.exists():
            utilities.add('TestClient')

        # Check for testing module (common pattern)
        testing_module = pkg_location / 'testing.py'
        if testing_module.exists():
            try:
                source = testing_module.read_text(encoding='utf-8', errors='ignore')
                test_tree = ast.parse(source)
                for node in ast.walk(test_tree):
                    if isinstance(node, ast.ClassDef) and 'Client' in node.name:
                        utilities.add(node.name)
            except (ValueError, TypeError, AttributeError, Exception):
                pass

        return list(utilities)[:10]  # Limit to 10

    @staticmethod
    def _extract_base_classes(tree: ast.AST) -> List[str]:
        """
        Extract important base classes

        These are classes users typically inherit from:
        - BaseModel (Pydantic)
        - TestCase (unittest)
        - APITestCase (Django REST)
        """
        base_classes = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Look for "Base" classes
                if node.name.startswith('Base') or node.name.endswith('Base'):
                    base_classes.add(node.name)

                # Look for common patterns
                if any(pattern in node.name for pattern in ['TestCase', 'Model', 'Schema']):
                    base_classes.add(node.name)

        return list(base_classes)[:10]  # Limit to 10

    @staticmethod
    def _extract_main_classes(tree: ast.AST) -> List[str]:
        """Extract main classes from package"""
        classes = []

        for node in cast(ast.Module, tree).body:  # Only top-level
            if isinstance(node, ast.ClassDef):
                # Skip private classes
                if not node.name.startswith('_'):
                    classes.append(node.name)

        return classes[:15]  # Limit to top 15

    @staticmethod
    def _extract_main_functions(tree: ast.AST) -> List[str]:
        """Extract main functions from package"""
        functions = []

        for node in cast(ast.Module, tree).body:  # Only top-level
            if isinstance(node, ast.FunctionDef):
                # Skip private functions
                if not node.name.startswith('_'):
                    functions.append(node.name)

        return functions[:15]  # Limit to top 15

    @staticmethod
    def _check_async_support(tree: ast.AST) -> bool:
        """Check if package has async support"""
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                return True

        return False
