# tests/unit/analysis/test_venv_scanner.py

"""
Tests for virtual environment scanner
"""

from pathlib import Path

from codecontext.analysis.venv_scanner import VirtualEnvScanner, PackageInfo
from codecontext.analysis.package_cache import PackageCache


class TestPackageInfo:
    """Tests for PackageInfo dataclass"""

    def test_to_dict(self):
        """Test conversion to dictionary"""
        info = PackageInfo(
            name="test-package",
            version="1.0.0",
            location=Path("/path/to/package"),
            test_utilities=["TestClient"],
            decorators=["app.route"],
            base_classes=["BaseModel"],
            async_support=True,
            main_classes=["MyClass"],
            main_functions=["my_function"],
        )

        data = info.to_dict()

        assert data["name"] == "test-package"
        assert data["version"] == "1.0.0"
        assert data["location"] == "/path/to/package"
        assert data["test_utilities"] == ["TestClient"]
        assert data["async_support"] is True

    def test_from_dict(self):
        """Test creation from dictionary"""
        data = {
            "name": "test-package",
            "version": "1.0.0",
            "location": "/path/to/package",
            "test_utilities": ["TestClient"],
            "decorators": ["app.route"],
            "base_classes": ["BaseModel"],
            "async_support": True,
            "main_classes": ["MyClass"],
            "main_functions": ["my_function"],
        }

        info = PackageInfo.from_dict(data)

        assert info.name == "test-package"
        assert info.version == "1.0.0"
        assert isinstance(info.location, Path)
        assert info.test_utilities == ["TestClient"]


class TestVirtualEnvScanner:
    """Tests for VirtualEnvScanner"""

    def test_detect_venv(self):
        """Test virtual environment detection"""
        scanner = VirtualEnvScanner()

        # Should detect something (current env or common locations)
        venv_path = scanner._detect_venv()

        # Maybe None if not in a venv, which is OK
        assert venv_path is None or isinstance(venv_path, Path)

    def test_find_site_packages(self):
        """Test finding site-packages directory"""
        scanner = VirtualEnvScanner()

        site_packages = scanner._find_site_packages()

        # Should find site-packages (even if not in venv)
        assert site_packages is None or isinstance(site_packages, Path)
        if site_packages:
            assert site_packages.exists()

    def test_scan_returns_dict(self):
        """Test scan returns dictionary of packages"""
        scanner = VirtualEnvScanner()

        # Scan with limit to make test fast
        packages = scanner.scan(limit=5)

        assert isinstance(packages, dict)

        # Should find at least some packages
        # (unless running in very minimal environment)
        if packages:
            for name, info in packages.items():
                assert isinstance(name, str)
                assert isinstance(info, PackageInfo)
                assert info.name == name
                assert isinstance(info.version, str)


class TestPackageCache:
    """Tests for package cache"""

    def test_save_and_load(self, tmp_path):
        """Test saving and loading cache"""
        cache_dir = tmp_path / "cache"
        cache = PackageCache(cache_dir)

        # Create test package info
        packages = {
            "test-pkg": PackageInfo(
                name="test-pkg",
                version="1.0.0",
                location=Path("/fake/path"),
                test_utilities=["TestClient"],
            )
        }

        # Save
        cache.save(packages, tmp_path)

        # Load
        loaded = cache.load()

        assert "test-pkg" in loaded
        assert loaded["test-pkg"].name == "test-pkg"
        assert loaded["test-pkg"].version == "1.0.0"
        assert loaded["test-pkg"].test_utilities == ["TestClient"]

    def test_should_refresh_no_cache(self, tmp_path):
        """Test should refresh when no cache exists"""
        cache_dir = tmp_path / "cache"
        cache = PackageCache(cache_dir)

        assert cache.should_refresh(tmp_path) is True

    def test_should_refresh_deps_changed(self, tmp_path):
        """Test should refresh when any of the dependencies change"""
        cache_dir = tmp_path / "cache"
        cache = PackageCache(cache_dir)

        # Create initial cache
        packages = {
            "test": PackageInfo(
                name="test",
                version="1.0.0",
                location=Path("/fake"),
            )
        }
        cache.save(packages, tmp_path)

        # Should not need refresh
        assert cache.should_refresh(tmp_path) is False

        # Modify requirements.txt
        requirements = tmp_path / "requirements.txt"
        requirements.write_text("new-package==1.0.0\n")

        # Should need refresh now
        assert cache.should_refresh(tmp_path) is True

    def test_clear(self, tmp_path):
        """Test clearing cache"""
        cache_dir = tmp_path / "cache"
        cache = PackageCache(cache_dir)

        # Create cache
        packages = {"test": PackageInfo(name="test", version="1.0", location=Path("/fake"))}
        cache.save(packages, tmp_path)

        assert cache.cache_file.exists()

        # Clear
        cache.clear()

        assert not cache.cache_file.exists()
