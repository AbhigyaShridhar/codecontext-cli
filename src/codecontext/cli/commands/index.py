"""
Index command - index codebase into vector store
"""

import click
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from codecontext.core.parser import CodebaseParser
from codecontext.core.config import load_config
from codecontext.search.vectorstore import MongoDBVectorStore

console = Console()


@click.command()
@click.option(
    '--path',
    type=click.Path(exists=True),
    default='.',
    help='Path to project root (default: current directory)'
)
@click.option(
    '--config',
    type=click.Path(exists=True),
    default=None,
    help='Path to config file (default: .codecontext.toml)'
)
def index(path: str, config: str) -> None:
    """
    Index codebase into vector store

    This parses your codebase and generates embeddings for semantic search.
    Run this after making significant code changes.

    Example:
        codecontext index
        codecontext index --path /path/to/project
    """
    project_root = Path(path).resolve()

    console.print(f"\n[bold cyan]Indexing codebase:[/bold cyan] {project_root}")
    console.print()

    # Load config
    cfg = load_config(project_root, config_path=config)

    # Parse codebase
    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
    ) as progress:
        task = progress.add_task("Parsing codebase...", total=None)

        parser = CodebaseParser(
            root=project_root,
            ignore_patterns=cfg.parsing.ignore_patterns
        )
        codebase = parser.parse()

        progress.update(task, description="✓ Parsing complete")

    console.print(f"  Files: {codebase.total_files}")
    console.print(f"  Functions: {codebase.total_functions}")
    console.print(f"  Classes: {codebase.total_classes}")
    console.print()

    # Connect to vector store
    console.print("[bold cyan]Connecting to vector store...[/bold cyan]")

    store = MongoDBVectorStore(
        connection_string=cfg.vector.connection_string,
        database_name=cfg.vector.database,
        collection_name=cfg.vector.collection,
        api_key=cfg.llm.openai_api_key
    )

    console.print("  ✓ Connected")
    console.print()

    # Index
    console.print("[bold cyan]Indexing (generating embeddings)...[/bold cyan]")

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
    ) as progress:
        task = progress.add_task("Generating embeddings...", total=codebase.total_functions + codebase.total_classes)

        def progress_callback(current, total):
            progress.update(task, completed=current, total=total)

        store.index_codebase(codebase, progress_callback=progress_callback)

        progress.update(task, description="✓ Indexing complete")

    # Show stats
    stats = store.get_stats()
    console.print()
    console.print("[bold green]✓ Indexing complete![/bold green]")
    console.print(f"  Total items: {stats['total_items']}")
    console.print(f"  Functions: {stats['functions']}")
    console.print(f"  Classes: {stats['classes']}")
    console.print()

    store.close()
