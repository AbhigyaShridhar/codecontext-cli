import ast
from typing import List


def get_node_source(node: ast.AST, source_lines: List[str]) -> str:
    """
    Get source code for an AST node

    Args:
        node: AST node
        source_lines: List of source code lines

    Returns:
        Source code as string
    """
    if not hasattr(node, 'lineno') or not hasattr(node, 'end_lineno'):
        return ""

    start = node.lineno - 1
    end = node.end_lineno

    if start < 0 or end > len(source_lines):
        return ""

    return "\n".join(source_lines[start:end])


def is_docstring(node: ast.AST) -> bool:
    """
    Check if node is a docstring

    Args:
        node: AST node

    Returns:
        True if node is a docstring
    """
    return (
            isinstance(node, ast.Expr) and
            isinstance(node.value, ast.Constant) and
            isinstance(node.value.value, str)
    )


def count_nodes_of_type(tree: ast.AST, node_type: type) -> int:
    """
    Count nodes of a specific type in AST

    Args:
        tree: AST tree
        node_type: Type of node to count (e.g., ast.If)

    Returns:
        Count of nodes

    Example:
        >>> node_tree = ast.parse("if x: pass\\nif y: pass")
        >>> count_nodes_of_type(node_tree, ast.If)
        2
    """
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, node_type):
            count += 1
    return count
