"""
CodeContext CLI

AI-powered test generation for Python projects.
"""

import click
from rich.console import Console

from codecontext.__version__ import __version__
from codecontext.cli.commands.init import init
from codecontext.cli.commands.index import index
from codecontext.cli.commands.ask import ask
from codecontext.cli.commands.generate import generate

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="codecontext")
def cli():
    """
    CodeContext - AI-powered test generation

    Learn your codebase patterns and generate contextually appropriate tests.
    """
    pass


# Register commands
# noinspection PyTypeChecker
cli.add_command(init)
# noinspection PyTypeChecker
cli.add_command(index)
# noinspection PyTypeChecker
cli.add_command(ask)
# noinspection PyTypeChecker
cli.add_command(generate)

if __name__ == '__main__':
    cli()
