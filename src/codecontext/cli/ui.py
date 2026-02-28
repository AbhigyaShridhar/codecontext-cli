from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich.syntax import Syntax
from typing import Optional

# Shared console instance
console = Console()


def print_header(text: str):
    """
    Print a formatted header
    """
    console.print(Panel.fit(
        f"[bold cyan]{text}[/bold cyan]",
        border_style="cyan"
    ))


def print_success(text: str):
    """Print success message"""
    console.print(f"[bold green]✓[/bold green] {text}")


def print_error(text: str):
    """Print error message"""
    console.print(f"[bold red]✗[/bold red] {text}")


def print_warning(text: str):
    """Print warning message"""
    console.print(f"[bold yellow]⚠[/bold yellow]  {text}")


def print_info(text: str):
    """Print info message"""
    console.print(f"[bold blue]ℹ[/bold blue]  {text}")


def print_code(code: str, language: str = "python", theme: str = "monokai"):
    """
    Print syntax-highlighted code

    Args:
        code: Source code to display
        language: Language for syntax highlighting
        theme: Color theme
    """
    syntax = Syntax(code, language, theme=theme, line_numbers=True)
    console.print(syntax)


def print_markdown(text: str):
    """
    Print formatted markdown

    Supports:
    - Headers
    - Lists
    - Code blocks
    - Bold/italic
    """
    md = Markdown(text)
    console.print(md)


def create_table(
        title: str,
        columns: list[str],
        rows: list[list[str]],
        show_header: bool = True
) -> Table:
    """
    Create a formatted table

    Args:
        title: Table title
        columns: Column headers
        rows: List of rows (each row is list of strings)
        show_header: Whether to show header row

    Returns:
        Rich Table object (use console.print(table))
    """
    table = Table(title=title, show_header=show_header)

    # Add columns
    for col in columns:
        table.add_column(col, style="cyan")

    # Add rows
    for row in rows:
        table.add_row(*row)

    return table


class ProgressBar:
    """
    Context manager for progress bars
    """

    def __init__(self, description: str):
        self.description = description
        self.progress = None
        self.task = None

    def __enter__(self):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        )
        self.progress.start()
        self.task = self.progress.add_task(self.description, total=None)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress.stop()

    def update(self, current: int, total: Optional[int] = None):
        """Update progress"""
        if total:
            self.progress.update(self.task, completed=current, total=total)
        else:
            self.progress.update(self.task, advance=1)


def confirm(message: str, default: bool = False) -> bool:
    """
    Ask for yes/no confirmation

    Args:
        message: Question to ask
        default: Default answer if user just presses Enter

    Returns:
        True if yes, False if no

    Example:
        >>> if confirm("Delete all files?", default=False):
        ...     # perform delete action
        ...     pass
    """
    import questionary

    return questionary.confirm(
        message,
        default=default
    ).ask()
