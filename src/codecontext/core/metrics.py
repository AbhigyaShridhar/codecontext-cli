"""
Code complexity metrics calculator

Calculates:
- Cyclomatic complexity (McCabe)
- Cognitive complexity (SonarSource)
- Maintainability index

These metrics help prioritize which code needs tests most.
"""

import ast


def calculate_cyclomatic_complexity(node: ast.FunctionDef) -> int:
    """
    Calculate cyclomatic complexity for a function

    Cyclomatic complexity = number of decision points + 1

    Decision points:
    - if, elif (each adds 1)
    - for, while (each adds 1)
    - except handlers (each adds 1)
    - boolean operators (and, or) in conditions (each adds 1)
    - comprehensions (each adds 1)

    Why this matters:
    - Higher complexity = harder to test
    - Complexity > 10 = probably too complex
    - Helps prioritize refactoring

    Args:
        node: AST node of the function

    Returns:
        Cyclomatic complexity score
    """
    complexity = 1  # Base complexity

    for child in ast.walk(node):
        # Conditional statements
        if isinstance(child, ast.If):
            complexity += 1

        # Loops
        elif isinstance(child, (ast.For, ast.While)):
            complexity += 1

        # Exception handlers
        elif isinstance(child, ast.ExceptHandler):
            complexity += 1

        # Comprehensions (list, dict, set, generator)
        elif isinstance(child, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
            complexity += 1

        # Boolean operators in conditions
        elif isinstance(child, ast.BoolOp):
            # Each 'and'/'or' adds complexity
            complexity += len(child.values) - 1

        # Ternary operator (x if condition else y)
        elif isinstance(child, ast.IfExp):
            complexity += 1

    return complexity


def calculate_cognitive_complexity(node: ast.FunctionDef) -> int:
    """
    Calculate cognitive complexity for a function

    Cognitive complexity measures "how hard is this to understand?"
    More accurate than cyclomatic for perceived difficulty.

    Rules (simplified version of SonarSource's algorithm):
    1. Each nesting level increases complexity
    2. Structural keywords (if, for, while) add 1
    3. Boolean operators add 1
    4. Recursion adds 1
    5. Breaks in linear flow add 1

    Cognitive complexity better reflects the actual difficulty of the code logic

    Args:
        node: AST _node of the function

    Returns:
        Cognitive complexity score
    """
    complexity = 0
    nesting_level = 0

    def visit(_node: ast.AST, level: int) -> int:
        """Recursively calculate the complexity with nesting awareness"""
        nonlocal complexity
        local_complexity = 0

        # Structural keywords increase complexity by (1 + nesting level)
        if isinstance(_node, (ast.If, ast.While, ast.For)):
            local_complexity += 1 + level

        # Exception handlers
        elif isinstance(_node, ast.ExceptHandler):
            local_complexity += 1 + level

        # Boolean operators (each adds 1)
        elif isinstance(_node, ast.BoolOp):
            local_complexity += len(_node.values) - 1

        # Break/continue (interrupts linear flow)
        elif isinstance(_node, (ast.Break, ast.Continue)):
            local_complexity += 1

        # Recursion detection (function calling itself)
        elif isinstance(_node, ast.Call):
            if isinstance(_node.func, ast.Name):
                if hasattr(_node, 'func') and _node.func.id == getattr(_node, 'id', None):
                    local_complexity += 1

        complexity += local_complexity

        # Determine new nesting level for children
        if isinstance(_node, (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.With)):
            new_level = level + 1
        else:
            new_level = level

        # Visit children
        for _child in ast.iter_child_nodes(_node):
            visit(_child, new_level)

        return complexity

    # Start visiting from function body
    for child in node.body:
        visit(child, nesting_level)

    return complexity


def calculate_maintainability_index(source_code: str) -> float:
    """
    Calculate maintainability index for code

    The maintainability index is a compound metric:
    - Halstead volume (vocabulary and length)
    - Cyclomatic complexity
    - Lines of code

    Formula (simplified):
    MI = max(0, (171-5.2 * ln(V) - 0.23 * G - 16.2 * ln(L)) * 100 / 171)

    Where:
    - V = Halstead volume
    - G = Cyclomatic complexity (average)
    - L = Lines of code

    Result:
    - 100: Perfect maintainability
    - 85-100: Highly maintainable
    - 65-85: Moderately maintainable
    - 0-65: Difficult to maintain

    For simplicity, we use a simplified approximation.

    Args:
        source_code: Complete source code of file

    Returns:
        Maintainability index (0-100)
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return 0.0

    # Count lines of code (non-comment, non-blank)
    lines = [line for line in source_code.splitlines()
             if line.strip() and not line.strip().startswith('#')]
    loc = len(lines)

    if loc == 0:
        return 100.0

    # Calculate average cyclomatic complexity
    total_complexity = 0
    function_count = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            total_complexity += calculate_cyclomatic_complexity(node)
            function_count += 1

    avg_complexity = total_complexity / max(function_count, 1)

    # Simplified maintainability index
    # Based on LOC and complexity
    # Higher LOC = lower maintainability
    # Higher complexity = lower maintainability

    import math

    # Penalize for lines of code (logarithmic)
    loc_penalty = 5.2 * math.log(max(loc, 1))

    # Penalize for complexity
    complexity_penalty = 0.23 * avg_complexity

    # Calculate index (normalized to 0-100)
    mi = max(
        0.0, (171 - loc_penalty - complexity_penalty - 16.2 * math.log(max(loc, 1))) * 100 / 171,
    )

    return round(mi, 2)


def get_function_loc(node: ast.FunctionDef) -> int:
    """
    Get lines of code for a function

    Counts actual lines, not including:
    - Blank lines
    - Comment lines
    - Docstring

    Args:
        node: Function AST node

    Returns:
        Number of lines of code
    """
    if node.end_lineno is None:
        return 0

    # Total lines (including blanks)
    total_lines = node.end_lineno - node.lineno + 1

    # This is approximate - we don't have the source here
    # In practice, we'll calculate this in the parser
    # where we have access to source lines

    return total_lines


def analyze_function(node: ast.FunctionDef, source_lines: list[str]) -> dict:
    """
    Comprehensive analysis of a function

    Args:
        node: Function AST node
        source_lines: List of source code lines

    Returns:
        Dict with all metrics:
        {
            'cyclomatic_complexity': int,
            'cognitive_complexity': int,
            'lines_of_code': int,
            'complexity_rating': str # 'simple', 'moderate', 'complex', 'very_complex'
        }
    """
    cyclomatic = calculate_cyclomatic_complexity(node)
    cognitive = calculate_cognitive_complexity(node)

    # Count actual LOC (non-blank, non-comment)
    function_lines = source_lines[node.lineno - 1:node.end_lineno]
    loc = len([line for line in function_lines
               if line.strip() and not line.strip().startswith('#')])

    # Rate overall complexity
    if cyclomatic <= 4:
        rating = "simple"
    elif cyclomatic <= 7:
        rating = "moderate"
    elif cyclomatic <= 10:
        rating = "complex"
    else:
        rating = "very_complex"

    return {
        'cyclomatic_complexity': cyclomatic,
        'cognitive_complexity': cognitive,
        'lines_of_code': loc,
        'complexity_rating': rating,
    }
