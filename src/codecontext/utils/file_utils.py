from pathlib import Path
from typing import List
import fnmatch


def find_python_files(
        root: Path,
        ignore_patterns: List[str]
) -> List[Path]:
    """
    Find all Python files in directory

    Args:
        root: Root directory
        ignore_patterns: Glob patterns to ignore

    Returns:
        List of Python file paths
    """
    python_files = []

    for path in root.rglob("*.py"):
        # Check if should be ignored
        relative = path.relative_to(root)

        should_ignore = False
        for pattern in ignore_patterns:
            if matches_pattern(str(relative), pattern):
                should_ignore = True
                break

        if not should_ignore:
            python_files.append(path)

    return python_files


def matches_pattern(path: str, pattern: str) -> bool:
    """
    Check if the path matches glob pattern

    Args:
        path: Path string
        pattern: Glob pattern

    Returns:
        True if matches
    """
    # Normalize slashes
    path = path.replace('\\', '/')
    pattern = pattern.replace('\\', '/')

    # Directory pattern
    if pattern.endswith('/'):
        return path.startswith(pattern) or f'/{pattern}' in path

    # Glob pattern
    return fnmatch.fnmatch(path, pattern)


def get_relative_module_name(file_path: Path, root: Path) -> str:
    """
    Get module name from a file path

    Example:
        >>> get_relative_module_name(
        ...     Path("/project/src/myapp/auth.py"),
        ...     Path("/project/src")
        ... )
        'myapp.auth'

    Args:
        file_path: Path to Python file
        root: Project root

    Returns:
        Module name (dot-separated)
    """
    try:
        relative = file_path.relative_to(root)
        parts = list(relative.parts[:-1])  # Remove filename
        parts.append(relative.stem)  # Add name without .py
        return ".".join(parts)
    except ValueError:
        return file_path.stem
