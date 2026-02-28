"""
Generate command - generate tests for functions
"""

import click
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from codecontext.core.parser import CodebaseParser
from codecontext.core.config import load_config
from codecontext.search.vectorstore import MongoDBVectorStore
from codecontext.agents.base import TestGenerationAgent

console = Console()


@click.command()
@click.option(
    '--function',
    '-f',
    required=True,
    help='Function name to generate test for'
)
@click.option(
    '--file',
    'file_path',
    required=True,
    help='File containing the function'
)
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
    '--output',
    '-o',
    type=click.Path(),
    default=None,
    help='Output file (default: stdout)'
)
def generate(function: str, file_path: str, path: str, config: str, output: str) -> None:
    """
    Generate test for a function

    Uses AI to analyze your codebase and generate appropriate tests.

    Examples:
        codecontext generate -f login -file src/auth.py
        codecontext generate -f process_data -file utils.py -o tests/test_utils.py
    """
    project_root = Path(path).resolve()

    console.print()
    console.print(Panel(
        f"[bold]Function:[/bold] {function}\n[bold]File:[/bold] {file_path}",
        title="Generating Test",
        border_style="cyan"
    ))
    console.print()

    # Load config
    cfg = load_config(project_root, config_path=config)

    # Parse codebase
    console.print("[cyan]Parsing codebase...[/cyan]")
    parser = CodebaseParser(
        root=project_root,
        ignore_patterns=cfg.parsing.ignore_patterns
    )
    codebase = parser.parse()
    console.print("  ✓ Parsed")
    console.print()

    # Connect to vector store
    console.print("[cyan]Connecting to vector store...[/cyan]")
    store = MongoDBVectorStore(
        connection_string=cfg.vector.connection_string,
        database_name=cfg.vector.database,
        collection_name=cfg.vector.collection,
        api_key=cfg.llm.openai_api_key
    )
    console.print("  ✓ Connected")
    console.print()

    # Initialize agent
    console.print("[cyan]Initializing agent...[/cyan]")
    agent = TestGenerationAgent(
        codebase=codebase,
        vector_store=store,
        project_root=project_root,
        llm_config={
            "model": cfg.llm.model,
            "api_key": cfg.llm.anthropic_api_key,
            "temperature": cfg.llm.temperature,
        }
    )
    console.print("  ✓ Agent ready")
    console.print()

    # Generate test
    console.print("[cyan]Generating test...[/cyan]")
    console.print("[dim](This may take 30-60 seconds)[/dim]")
    console.print()

    try:
        test_code = agent.generate_test(
            target_function=function,
            target_file=file_path
        )

        # Display result
        console.print("[bold green]✓ Test generated![/bold green]")
        console.print()

        # Syntax highlight
        syntax = Syntax(test_code, "python", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title="Generated Test", border_style="green"))

        # Save to file if requested
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(test_code)
            console.print()
            console.print(f"[green]✓ Saved to:[/green] {output_path}")

        console.print()

    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {e}")
        raise

    finally:
        store.close()
