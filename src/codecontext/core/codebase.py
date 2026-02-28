from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Set
from datetime import datetime


class DirectedGraph:
    """
    Simple directed graph implementation

    Stores:
    - Nodes (with optional data)
    - Edges (directed: A -> B)

    Provides:
    - Add/query nodes and edges
    - Get successors (what A points to)
    - Get predecessors (what points to A)
    """

    def __init__(self):
        # Node data: {node_id: metadata}
        self.nodes: Dict[str, dict] = {}

        # Edges: {node_id: set of node_ids it points to}
        self.edges: Dict[str, Set[str]] = {}

        # Reverse edges: {node_id: set of node_ids that point to it, a reverse relationship}
        self.reverse_edges: Dict[str, Set[str]] = {}

    def add_node(self, node_id: str, **data):
        """
        Add a node with optional metadata
        """
        if node_id not in self.nodes:
            self.nodes[node_id] = data
            self.edges[node_id] = set()
            self.reverse_edges[node_id] = set()

    def add_edge(self, from_node: str, to_node: str):
        """
        Add a directed edge: from_node -> to_node
        """
        # Ensure nodes exist
        if from_node not in self.nodes:
            self.add_node(from_node)
        if to_node not in self.nodes:
            self.add_node(to_node)

        # Add edge
        self.edges[from_node].add(to_node)

        # Add reverse edge (for efficient predecessor queries)
        self.reverse_edges[to_node].add(from_node)

    def successors(self, node_id: str) -> List[str]:
        """
        Get all nodes this node points to
        """
        if node_id not in self.edges:
            return []
        return list(self.edges[node_id])

    def predecessors(self, node_id: str) -> List[str]:
        """
        Get all nodes that point to this node
        """
        if node_id not in self.reverse_edges:
            return []
        return list(self.reverse_edges[node_id])

    def has_node(self, node_id: str) -> bool:
        """Check if the node exists"""
        return node_id in self.nodes

    def get_node_data(self, node_id: str) -> dict:
        """Get metadata for a node"""
        return self.nodes.get(node_id, {})

    def all_nodes(self) -> List[str]:
        """Get all node IDs"""
        return list(self.nodes.keys())

    def node_count(self) -> int:
        """Get number of nodes"""
        return len(self.nodes)

    def edge_count(self) -> int:
        """Get number of edges"""
        return sum(len(edges) for edges in self.edges.values())


@dataclass(frozen=True)
class FunctionInfo:
    """Represents a parsed Python function"""

    # Identity
    name: str
    file_path: Path
    line_start: int
    line_end: int

    # Code
    signature: str
    code: str
    docstring: Optional[str] = None

    # Metadata
    is_async: bool = False
    is_method: bool = False
    is_staticmethod: bool = False
    is_classmethod: bool = False
    is_property: bool = False

    # Dependencies
    calls: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)

    # Metrics
    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0
    lines_of_code: int = 0

    # Type information
    parameters: Dict[str, Optional[str]] = field(default_factory=dict)
    return_type: Optional[str] = None

    # Decorators
    decorators: List[str] = field(default_factory=list)

    def __hash__(self):
        return hash((str(self.file_path), self.name, self.line_start))

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "file_path": str(self.file_path),
            "line_start": self.line_start,
            "line_end": self.line_end,
            "signature": self.signature,
            "code": self.code,
            "docstring": self.docstring,
            "is_async": self.is_async,
            "is_method": self.is_method,
            "is_staticmethod": self.is_staticmethod,
            "is_classmethod": self.is_classmethod,
            "is_property": self.is_property,
            "calls": self.calls,
            "imports": self.imports,
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "cognitive_complexity": self.cognitive_complexity,
            "lines_of_code": self.lines_of_code,
            "parameters": self.parameters,
            "return_type": self.return_type,
            "decorators": self.decorators,
        }


@dataclass(frozen=True)
class ClassInfo:
    """Represents a parsed Python class"""

    # Identity
    name: str
    file_path: Path
    line_start: int
    line_end: int

    # Code
    code: str
    docstring: Optional[str] = None

    # Inheritance
    bases: List[str] = field(default_factory=list)

    # Members
    methods: List[FunctionInfo] = field(default_factory=list)
    properties: List[str] = field(default_factory=list)
    class_variables: List[str] = field(default_factory=list)

    # Decorators
    decorators: List[str] = field(default_factory=list)

    def __hash__(self):
        return hash((str(self.file_path), self.name, self.line_start))

    def get_method(self, name: str) -> Optional[FunctionInfo]:
        """Get a method by name"""
        for method in self.methods:
            if method.name == name:
                return method
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "file_path": str(self.file_path),
            "line_start": self.line_start,
            "line_end": self.line_end,
            "code": self.code,
            "docstring": self.docstring,
            "bases": self.bases,
            "methods": [m.to_dict() for m in self.methods],
            "properties": self.properties,
            "class_variables": self.class_variables,
            "decorators": self.decorators,
        }


@dataclass(frozen=True)
class FileInfo:
    """Represents a parsed Python file"""

    # Identity
    path: Path
    module_name: str

    # Metadata
    docstring: Optional[str] = None
    lines_of_code: int = 0
    last_modified: float = 0

    # Contents
    functions: List[FunctionInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)

    # Metrics
    maintainability_index: float = 0.0

    def __hash__(self):
        return hash(str(self.path))

    def get_all_functions(self) -> List[FunctionInfo]:
        """Get all functions (top-level + class methods)"""
        all_funcs = list(self.functions)

        for cls in self.classes:
            all_funcs.extend(cls.methods)

        return all_funcs

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "path": str(self.path),
            "module_name": self.module_name,
            "docstring": self.docstring,
            "lines_of_code": self.lines_of_code,
            "last_modified": self.last_modified,
            "functions": [f.to_dict() for f in self.functions],
            "classes": [c.to_dict() for c in self.classes],
            "imports": self.imports,
            "maintainability_index": self.maintainability_index,
        }


@dataclass
class Codebase:
    """
    Complete representation of a parsed codebase
    """

    # Raw parsed data
    files: Dict[Path, FileInfo]

    # Indexes for fast lookup
    functions_by_name: Dict[str, List[FunctionInfo]]
    classes_by_name: Dict[str, List[ClassInfo]]

    # Dependency graphs (using our DirectedGraph!)
    call_graph: DirectedGraph  # Function -> Function calls
    import_graph: DirectedGraph  # Module -> Module imports

    # Statistics
    total_files: int = 0
    total_functions: int = 0
    total_classes: int = 0
    total_lines: int = 0

    # Metadata
    parsed_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_files(cls, files: Dict[Path, FileInfo]) -> "Codebase":
        """Build a Codebase from parsed files"""

        # Build indexes
        functions_by_name: Dict[str, List[FunctionInfo]] = {}
        classes_by_name: Dict[str, List[ClassInfo]] = {}

        total_functions = 0
        total_classes = 0
        total_lines = 0

        for file_info in files.values():
            total_lines += file_info.lines_of_code

            # Index functions
            for func in file_info.get_all_functions():
                total_functions += 1
                if func.name not in functions_by_name:
                    functions_by_name[func.name] = []
                functions_by_name[func.name].append(func)

            # Index classes
            for _cls in file_info.classes:
                total_classes += 1
                if _cls.name not in classes_by_name:
                    classes_by_name[_cls.name] = []
                classes_by_name[_cls.name].append(_cls)

        # Build graphs (using our DirectedGraph!)
        call_graph = cls._build_call_graph(files)
        import_graph = cls._build_import_graph(files)

        return cls(
            files=files,
            functions_by_name=functions_by_name,
            classes_by_name=classes_by_name,
            call_graph=call_graph,
            import_graph=import_graph,
            total_files=len(files),
            total_functions=total_functions,
            total_classes=total_classes,
            total_lines=total_lines,
        )

    @staticmethod
    def _build_call_graph(files: Dict[Path, FileInfo]) -> DirectedGraph:
        """Build function call dependency graph"""

        graph = DirectedGraph()

        for file_info in files.values():
            for func in file_info.get_all_functions():
                # Create unique ID for this function
                func_id = f"{func.file_path}:{func.name}"

                # Add node with metadata
                graph.add_node(func_id, function=func)

                # Add edges for calls
                for called in func.calls:
                    graph.add_edge(func_id, called)

        return graph

    @staticmethod
    def _build_import_graph(files: Dict[Path, FileInfo]) -> DirectedGraph:
        """Build module import dependency graph"""

        graph = DirectedGraph()

        for file_info in files.values():
            module_id = file_info.module_name

            # Add node
            graph.add_node(module_id, file=file_info)

            # Add edges for imports
            for imported in file_info.imports:
                graph.add_edge(module_id, imported)

        return graph

    def find_function(self, name: str) -> Optional[List[FunctionInfo]]:
        """Find functions by name"""
        return self.functions_by_name.get(name)

    def find_class(self, name: str) -> Optional[List[ClassInfo]]:
        """Find classes by name"""
        return self.classes_by_name.get(name)

    def get_function_dependencies(self, func: FunctionInfo) -> List[FunctionInfo]:
        """Get all functions this function calls"""
        func_id = f"{func.file_path}:{func.name}"

        if not self.call_graph.has_node(func_id):
            return []

        dependencies = []

        # Get all functions this one calls
        for called_id in self.call_graph.successors(func_id):
            # Try to find the actual function
            called_name = called_id.split(":")[-1] if ":" in called_id else called_id
            if called_name in self.functions_by_name:
                dependencies.extend(self.functions_by_name[called_name])

        return dependencies

    def get_function_dependents(self, func: FunctionInfo) -> List[FunctionInfo]:
        """Get all functions that call this one"""
        func_id = f"{func.file_path}:{func.name}"

        if not self.call_graph.has_node(func_id):
            return []

        dependents = []

        # Get all functions that call this one
        for caller_id in self.call_graph.predecessors(func_id):
            caller_name = caller_id.split(":")[-1] if ":" in caller_id else caller_id
            if caller_name in self.functions_by_name:
                dependents.extend(self.functions_by_name[caller_name])

        return dependents

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "files": {
                str(path): file_info.to_dict()
                for path, file_info in self.files.items()
            },
            "total_files": self.total_files,
            "total_functions": self.total_functions,
            "total_classes": self.total_classes,
            "total_lines": self.total_lines,
            "parsed_at": self.parsed_at.isoformat(),
        }

    def get_statistics(self) -> dict:
        """Get codebase statistics"""
        return {
            "total_files": self.total_files,
            "total_functions": self.total_functions,
            "total_classes": self.total_classes,
            "total_lines": self.total_lines,
            "avg_lines_per_file": self.total_lines / max(self.total_files, 1),
            "avg_functions_per_file": self.total_functions / max(self.total_files, 1),
        }
