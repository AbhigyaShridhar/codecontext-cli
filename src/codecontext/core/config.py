from pathlib import Path
from typing import Optional, Literal, List
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
import yaml
import os


class LLMConfig(BaseModel):
    """
    Configuration for Large Language Model

    Attributes:
        provider: LLM provider (currently only 'anthropic' supported)
        model: Model identifier (e.g., 'claude-sonnet-4-20250514')
        api_key: API key (loaded from env if not specified)
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
    """
    provider: Literal["anthropic"] = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    api_key: Optional[str] = None
    max_tokens: int = 4000
    temperature: float = 0.7

    @classmethod
    @field_validator("temperature")
    def validate_temperature(cls, v: float) -> float:
        """Ensure temperature is in valid range"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")
        return v

    @classmethod
    @field_validator("max_tokens")
    def validate_max_tokens(cls, v: int) -> int:
        """Ensure max_tokens is reasonable"""
        if v < 100 or v > 200000:
            raise ValueError("max_tokens must be between 100 and 200000")
        return v

    def get_api_key(self) -> str:
        """
        Get the API key from config or environment

        Priority:
        1. Explicitly set api_key
        2. ANTHROPIC_API_KEY environment variable

        Raises:
            ValueError: If no API key found
        """
        if self.api_key:
            return self.api_key

        env_key = os.getenv("ANTHROPIC_API_KEY")
        if env_key:
            return env_key

        raise ValueError(
            "No Anthropic API key found. Set ANTHROPIC_API_KEY environment "
            "variable or specify in config."
        )


class SearchConfig(BaseModel):
    """
    Configuration for code search

    Attributes:
        mode: Search mode (auto/keyword/semantic/hybrid)
        smart_warnings: Whether to warn about search limitations
        suggest_vector_db: Whether to suggest enabling vector DB
    """

    mode: Literal["auto", "keyword", "semantic", "hybrid"] = "auto"
    smart_warnings: bool = True
    suggest_vector_db: bool = True

    class Config:
        # Pydantic v2 config
        validate_assignment = True


class ProjectConfig(BaseModel):
    """
    Project-specific configuration

    Attributes:
        name: Project name
        root: Project root directory (must exist)
        ignore_patterns: Patterns to ignore when scanning
        test_framework: Testing framework to use
    """

    name: str = Field(description="Project name")
    root: Path = Field(description="Project root directory")
    ignore_patterns: List[str] = Field(
        default_factory=lambda: [
            # Virtual environments
            "venv/", "env/", ".venv/", "ENV/",
            # Dependencies
            "node_modules/",
            # Git
            ".git/",
            # Python artifacts
            "__pycache__/", "*.pyc", "*.pyo", "*.pyd",
            # Build artifacts
            "dist/", "build/", "*.egg-info/",
            # Testing
            ".pytest_cache/", ".mypy_cache/", ".tox/",
            # Coverage
            ".coverage", "htmlcov/",
        ],
        description="Patterns to ignore when parsing"
    )
    test_framework: Literal["pytest", "unittest"] = "pytest"

    @classmethod
    @field_validator("root")
    def validate_root_exists(cls, v: Path) -> Path:
        """Ensure project root exists"""
        if not v.exists():
            raise ValueError(f"Project root does not exist: {v}")
        if not v.is_dir():
            raise ValueError(f"Project root is not a directory: {v}")
        return v.resolve()  # Convert to absolute path

    @classmethod
    @field_validator("root", mode="before")
    def convert_to_path(cls, v) -> Path:
        """Convert string to Path if needed"""
        if isinstance(v, str):
            return Path(v)
        return v

    venv_path: Optional[Path] = Field(
        default=None,
        description="Path to virtual environment (auto-detected if None)"
    )

    @classmethod
    @field_validator("root")
    def validate_root_exists(cls, v: Path) -> Path:
        """Ensure project root exists"""
        if not v.exists():
            raise ValueError(f"Project root does not exist: {v}")
        return v.resolve()

    model_config = ConfigDict(validate_assignment=True)


class VectorConfig(BaseModel):
    """
    Vector store configuration

    Settings for MongoDB Atlas vector search
    """
    enabled: bool = True
    storage_provider: Literal["mongodb"] = "mongodb"

    # MongoDB settings
    mongodb_connection_string: Optional[str] = Field(
        default=None,
        description="MongoDB Atlas connection string"
    )
    database_name: str = Field(
        default="codecontext",
        description="Database name in MongoDB"
    )
    collection_name: str = Field(
        default="codebase",
        description="Collection name for code embeddings"
    )

    # Embedding settings
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model"
    )
    embedding_dimensions: int = Field(
        default=1536,
        description="Embedding vector dimensions"
    )
    batch_size: int = Field(
        default=100,
        description="Batch size for embedding generation"
    )

    def get_connection_string(self) -> Optional[str]:
        """
        Get MongoDB connection string

        Priority:
        1. Explicitly set connection string
        2. MONGODB_URI environment variable
        """
        if self.mongodb_connection_string:
            return self.mongodb_connection_string

        import os
        return os.getenv("MONGODB_URI")

    model_config = ConfigDict(validate_assignment=True)


class CodeContextConfig(BaseModel):
    """
    Complete CodeContext configuration

    This is the main config object used throughout the application.

    Example:
        >>> config = CodeContextConfig(
        ...     project=ProjectConfig(
        ...         name="my-project",
        ...         root=Path("/path/to/project")
        ...     )
        ... )
        >>> print(config.llm.model)
        'claude-sonnet-4-20250514'
    """

    # Version of config format (for future migrations)
    version: str = "1.0.0"

    # Main configuration sections
    project: ProjectConfig
    llm: LLMConfig = Field(default_factory=LLMConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    vectordb: VectorConfig = Field(default_factory=VectorConfig)

    # =================================================================
    # CLASS METHODS - Alternative Constructors
    # =================================================================

    @classmethod
    def from_yaml(cls, path: Path) -> "CodeContextConfig":
        """
        Load configuration from YAML file

        Args:
            path: Path to YAML config file

        Returns:
            CodeContextConfig instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If YAML is invalid

        Example:
            >>> config = CodeContextConfig.from_yaml(
            ...     Path(".codecontext/config.yaml")
            ... )
        """
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        try:
            with open(path) as f:
                data = yaml.safe_load(f)

            # YAML might have None instead of empty dict
            if data is None:
                raise ValueError("Config file is empty")

            return cls(**data)

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {path}: {e}")

    @classmethod
    def from_project_root(cls, project_root: Path) -> "CodeContextConfig":
        """
        Load config from project's .codecontext directory

        Looks for: {project_root}/.codecontext/config.yaml

        Args:
            project_root: Project root directory

        Returns:
            CodeContextConfig instance

        Raises:
            FileNotFoundError: If config doesn't exist
        """
        config_path = project_root / ".codecontext" / "config.yaml"
        return cls.from_yaml(config_path)

    # =================================================================
    # INSTANCE METHODS
    # =================================================================

    def to_yaml(self, path: Path) -> None:
        """
        Save configuration to YAML file

        Args:
            path: Where to save config

        Example:
            config.to_yaml(Path(".codecontext/config.yaml"))
        """
        # Create parent directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict (Pydantic method)
        data = self.model_dump(
            mode="json",  # Use JSON-compatible types
            exclude_none=True,  # Don't include None values
        )

        # Write YAML
        with open(path, 'w') as f:
            yaml.dump(
                data,
                f,
                default_flow_style=False,  # Use block style (more readable)
                sort_keys=False,  # Keep order from dict
            )

    def get_config_dir(self) -> Path:
        """
        Get .codecontext directory for this project

        Returns:
            Path to .codecontext/ directory
        """
        return self.project.root / ".codecontext"

    def get_cache_dir(self) -> Path:
        """
        Get cache directory for this project

        Returns:
            Path to .codecontext/cache/ directory
        """
        return self.get_config_dir() / "cache"

    # =================================================================
    # VALIDATION
    # =================================================================

    @model_validator(mode="after")
    def validate_consistency(self):
        """Validate configuration consistency"""

        # If semantic search requested, ensure vector DB enabled
        if self.search.mode in ["semantic", "hybrid"]:
            if not self.vectordb.enabled:
                # Warn but don't fail (will fall back to keyword)
                pass

        return self

    def get_vector_storage_path(self) -> Path:
        """Get the path for vector store data"""
        return self.get_config_dir() / "vectors"

    def get_cache_path(self) -> Path:
        """Get the path for cached data"""
        return self.get_config_dir() / "cache"

    class Config:
        # Pydantic v2 config
        validate_assignment = True
        use_enum_values = True
        extra = "forbid"  # Fail on unknown keys


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _dict_to_namespace(d: dict):
    """Recursively convert a dict to SimpleNamespace for attribute-style access."""
    from types import SimpleNamespace
    ns = SimpleNamespace()
    for key, value in d.items():
        setattr(ns, key, _dict_to_namespace(value) if isinstance(value, dict) else value)
    return ns


def _load_toml_config(path: Path):
    """Parse a TOML config file and return a SimpleNamespace object."""
    import sys
    if sys.version_info >= (3, 11):
        import tomllib
        with open(path, "rb") as f:
            data = tomllib.load(f)
    else:
        import tomli
        with open(path, "rb") as f:
            data = tomli.load(f)
    return _dict_to_namespace(data)


def load_config(project_root: Optional[Path] = None, config_path: Optional[str] = None):
    """
    Convenience function to load config

    Looks for .codecontext.toml at the project root (created by 'codecontext init').

    Args:
        project_root: Project root directory (default: current directory)
        config_path: Explicit path to config file; overrides the default lookup

    Returns:
        SimpleNamespace with nested attributes matching the TOML structure
        (e.g. cfg.vector.connection_string, cfg.llm.openai_api_key)

    Example:
        >>> config = load_config(Path("/path/to/project"))
        >>> config = load_config(config_path="/custom/.codecontext.toml")
    """
    if config_path is not None:
        explicit_path = Path(config_path)
        if not explicit_path.exists():
            raise FileNotFoundError(
                f"Configuration not found at {explicit_path}\n"
                f"Run 'codecontext init' to initialize the project."
            )
        return _load_toml_config(explicit_path)

    if project_root is None:
        project_root = Path.cwd()

    config_file = project_root / ".codecontext.toml"
    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration not found at {config_file}\n"
            f"Run 'codecontext init' to initialize the project."
        )

    return _load_toml_config(config_file)


def create_default_config(
        project_root: Path,
        project_name: Optional[str] = None
) -> CodeContextConfig:
    """
    Create a default configuration

    Args:
        project_root: Project root directory
        project_name: Project name (default: directory name)

    Returns:
        CodeContextConfig with default values

    Example:
        >>> config = create_default_config(Path.cwd())
        >>> config.to_yaml(Path(".codecontext/config.yaml"))
    """
    if project_name is None:
        project_name = project_root.name

    return CodeContextConfig(
        project=ProjectConfig(
            name=project_name,
            root=project_root,
        ),
        llm=LLMConfig(),
        vector=VectorConfig()
    )
