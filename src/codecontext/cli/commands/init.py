"""
Init command - initialize CodeContext in a project
"""

import click
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt, Confirm

console = Console()


@click.command()
@click.option(
    '--path',
    type=click.Path(),
    default='.',
    help='Path to project root'
)
def init(path: str) -> None:
    """
    Initialize CodeContext in your project

    Creates a .codecontext.toml configuration file.

    Example:
        codecontext init
        codecontext init --path /path/to/project
    """
    project_root = Path(path).resolve()
    config_path = project_root / '.codecontext.toml'

    console.print()
    console.print("[bold cyan]CodeContext Initialization[/bold cyan]")
    console.print()

    if config_path.exists():
        if not Confirm.ask(f"[yellow]Config already exists at {config_path}. Overwrite?[/yellow]"):
            console.print("[yellow]Aborted.[/yellow]")
            return

    # Gather configuration
    console.print("[bold]Vector Store Configuration[/bold]")

    mongodb_uri = Prompt.ask(
        "MongoDB URI",
        default="mongodb://localhost:27017"
    )

    openai_key = Prompt.ask(
        "OpenAI API Key",
        password=True
    )

    console.print()
    console.print("[bold]LLM Configuration[/bold]")

    anthropic_key = Prompt.ask(
        "Anthropic API Key",
        password=True
    )

    model = Prompt.ask(
        "Model",
        default="claude-sonnet-4-20250514"
    )

    # Create config
    config_content = f"""# CodeContext Configuration

[parsing]
# Files/directories to ignore during parsing
ignore_patterns = [
    "*.pyc",
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
    "build",
    "dist",
]

[vector]
# Vector store configuration
connection_string = "{mongodb_uri}"
database = "codecontext"
collection = "codebase"

[llm]
# LLM configuration for test generation
model = "{model}"
temperature = 0.0

# API Keys (you can also set these as environment variables)
openai_api_key = "{openai_key}"
anthropic_api_key = "{anthropic_key}"

[generation]
# Test generation settings
max_retries = 3
timeout = 120
"""

    # Write config
    config_path.write_text(config_content)

    console.print()
    console.print(f"[bold green]✓ Configuration saved to:[/bold green] {config_path}")
    console.print()
    console.print("[bold]Next steps:[/bold]")
    console.print("  1. Review and edit .codecontext.toml")
    console.print("  2. Run: [cyan]codecontext index[/cyan]")
    console.print("  3. Try: [cyan]codecontext ask 'authentication functions'[/cyan]")
    console.print()
