"""
Ask command - search codebase using natural language
"""

import click
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from codecontext.core.config import load_config
from codecontext.search.vectorstore import MongoDBVectorStore

console = Console()


@click.command()
@click.argument('query')
@click.option(
    '--path',
    type=click.Path(exists=True),
    default='.',
    help='Path to project root'
)
@click.option(
    '--config',
    type=click.Path(exists=True),
    default=None,
    help='Path to config file'
)
@click.option(
    '--limit',
    '-n',
    type=int,
    default=5,
    help='Number of results (default: 5)'
)
@click.option(
    '--type',
    'filter_type',
    type=click.Choice(['function', 'class']),
    default=None,
    help='Filter by type'
)
def ask(query: str, path: str, config: str, limit: int, filter_type: str) -> None:
    """
    Search codebase using natural language

    Uses semantic search to find relevant code based on your query.

    Examples:
        codecontext ask "user authentication functions"
        codecontext ask "database connection handling" --limit 10
        codecontext ask "API endpoints" --type function
    """
    project_root = Path(path).resolve()

    # Load config
    cfg = load_config(project_root, config_path=config)

    console.print()
    console.print(Panel(f"[bold]Query:[/bold] {query}", border_style="cyan"))
    console.print()

    # Connect to vector store
    store = MongoDBVectorStore(
        connection_string=cfg.vector.connection_string,
        database_name=cfg.vector.database,
        collection_name=cfg.vector.collection,
        api_key=cfg.llm.openai_api_key
    )

    # Search
    try:
        results = store.search(query, k=limit, filter_type=filter_type)

        if not results:
            console.print("[yellow]No results found.[/yellow]")
            console.print()
            return

        # Display results
        console.print(f"[bold green]Found {len(results)} results:[/bold green]")
        console.print()

        for i, result in enumerate(results, 1):
            # Format result
            header = f"{i}. {result.name} ({result.type})"
            file_path = Path(result.file).relative_to(project_root) if project_root in Path(
                result.file).parents else result.file

            content = [f"[bold]File:[/bold] {file_path}", f"[bold]Similarity:[/bold] {result.score:.3f}"]

            if result.metadata.get('signature'):
                content.append(f"[bold]Signature:[/bold] `{result.metadata['signature']}`")

            if result.metadata.get('docstring'):
                docstring = result.metadata['docstring'][:200]
                content.append(f"\n{docstring}...")

            console.print(Panel(
                "\n".join(content),
                title=header,
                border_style="blue"
            ))
            console.print()

    finally:
        store.close()
