# src/codecontext/analysis/package_cache.py

"""
Cache scanned package information

Only re-scan when:
- pyproject.toml changes
- requirements.txt changes
- User runs 'codecontext refresh'
"""

import json
from pathlib import Path
from typing import Dict
import hashlib

from codecontext.analysis.venv_scanner import PackageInfo


class PackageCache:
    """
    Cache for scanned package information

    Prevents unnecessary re-scanning of virtual environment.
    """

    def __init__(self, cache_dir: Path):
        """
        Initialize cache

        Args:
            cache_dir: Directory to store cache files (e.g., .codecontext/cache/)
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_file = cache_dir / "venv_packages.json"
        self.deps_hash_file = cache_dir / "deps_hash.txt"

    def should_refresh(self, project_root: Path) -> bool:
        """
        Check if cache should be refreshed

        Refresh when:
        - Cache doesn't exist
        - Dependencies changed (pyproject.toml/requirements.txt modified)

        Args:
            project_root: Project root directory

        Returns:
            True if cache needs refresh
        """
        if not self.cache_file.exists():
            return True

        # Calculate current dependencies hash
        current_hash = self._hash_dependencies(project_root)

        # Compare with cached hash
        if not self.deps_hash_file.exists():
            return True

        try:
            cached_hash = self.deps_hash_file.read_text().strip()
            return current_hash != cached_hash
        except (ValueError, TypeError, AttributeError, Exception):
            return True

    @staticmethod
    def _hash_dependencies(project_root: Path) -> str:
        """
        Hash dependency files to detect changes

        Hashes:
        - pyproject.toml
        - requirements.txt
        - requirements-dev.txt
        - setup.py
        - Pipfile

        Args:
            project_root: Project root directory

        Returns:
            SHA256 hash of all dependency files
        """
        hasher = hashlib.sha256()

        # Files to check
        dep_files = [
            'pyproject.toml',
            'requirements.txt',
            'requirements-dev.txt',
            'setup.py',
            'Pipfile',
        ]

        for filename in dep_files:
            file_path = project_root / filename
            if file_path.exists():
                try:
                    hasher.update(file_path.read_bytes())
                except (ValueError, TypeError, AttributeError, Exception):
                    pass

        return hasher.hexdigest()

    def save(self, packages: Dict[str, PackageInfo], project_root: Path) -> None:
        """
        Save package info to cache

        Args:
            packages: Dict of package name -> PackageInfo
            project_root: Project root (to save dependency hash)
        """
        try:
            # Convert PackageInfo to dict
            cache_data = {}
            for name, info in packages.items():
                cache_data[name] = info.to_dict()

            # Save to file
            self.cache_file.write_text(json.dumps(cache_data, indent=2))

            # Save dependencies hash
            deps_hash = self._hash_dependencies(project_root)
            self.deps_hash_file.write_text(deps_hash)

        except Exception as e:
            # Cache save failed, but don't crash
            print(f"Warning: Failed to save package cache: {e}")

    def load(self) -> Dict[str, PackageInfo]:
        """
        Load package info from cache

        Returns:
            Dict of package name -> PackageInfo
            Empty dict if cache doesn't exist or is invalid
        """
        if not self.cache_file.exists():
            return {}

        try:
            data = json.loads(self.cache_file.read_text())

            # Convert back to PackageInfo objects
            packages = {}
            for name, info_dict in data.items():
                packages[name] = PackageInfo.from_dict(info_dict)

            return packages

        except (ValueError, TypeError, AttributeError, Exception):
            # Cache corrupted, return empty
            return {}

    def clear(self) -> None:
        """Clear the cache"""
        if self.cache_file.exists():
            self.cache_file.unlink()
        if self.deps_hash_file.exists():
            self.deps_hash_file.unlink()
