"""
CLI integration tests

Test CLI commands work end-to-end
"""

import pytest
import os
import tempfile
from pathlib import Path
from click.testing import CliRunner

from codecontext.cli.main import cli

pytestmark = pytest.mark.skipif(
    not os.getenv("MONGODB_URI") or not os.getenv("OPENAI_API_KEY") or not os.getenv("ANTHROPIC_API_KEY"),
    reason="Requires credentials"
)


@pytest.fixture
def sample_project():
    """Create sample project"""
    with tempfile.TemporaryDirectory() as tmp:
        project_root = Path(tmp)

        # Create simple Python file
        (project_root / "main.py").write_text("""
def hello(name: str) -> str:
    '''Greet someone by name'''
    return f"Hello, {name}!"

def add(a: int, b: int) -> int:
    '''Add two numbers'''
    return a + b
""")

        yield project_root


class TestCLIIntegration:
    """Test CLI commands"""

    def test_init_command(self, sample_project):
        """Test codecontext init"""
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=sample_project):
            # Run init with input
            result = runner.invoke(
                cli, ['init'],
                input=f"{os.getenv('MONGODB_URI')}\n{os.getenv('OPENAI_API_KEY')}\n{os.getenv('ANTHROPIC_API_KEY')}\nclaude-sonnet-4-20250514\n")

            assert result.exit_code == 0
            assert "Configuration saved" in result.output

            # Check config file created
            config_file = sample_project / ".codecontext.toml"
            assert config_file.exists()

            config_content = config_file.read_text()
            assert "codecontext" in config_content.lower()

    def test_index_command(self, sample_project):
        """Test codecontext index"""
        runner = CliRunner()

        # Create config first
        config_file = sample_project / ".codecontext.toml"
        config_file.write_text(f"""
[parsing]
ignore_patterns = ["__pycache__"]

[vector]
connection_string = "{os.getenv('MONGODB_URI')}"
database = "codecontext_cli_test"
collection = "test"

[llm]
model = "claude-sonnet-4-20250514"
temperature = 0.0
openai_api_key = "{os.getenv('OPENAI_API_KEY')}"
anthropic_api_key = "{os.getenv('ANTHROPIC_API_KEY')}"
""")

        # Run index
        result = runner.invoke(cli, ['index', '--path', str(sample_project)])

        assert result.exit_code == 0
        assert "Indexing complete" in result.output

    def test_ask_command(self, sample_project):
        """Test codecontext ask"""
        runner = CliRunner()

        # Setup (create config and index)
        config_file = sample_project / ".codecontext.toml"
        config_file.write_text(f"""
[parsing]
ignore_patterns = []

[vector]
connection_string = "{os.getenv('MONGODB_URI')}"
database = "codecontext_cli_test"
collection = "ask_test"

[llm]
openai_api_key = "{os.getenv('OPENAI_API_KEY')}"
anthropic_api_key = "{os.getenv('ANTHROPIC_API_KEY')}"
""")

        # Index first
        runner.invoke(cli, ['index', '--path', str(sample_project)])

        # Now ask
        result = runner.invoke(cli, ['ask', 'greeting functions', '--path', str(sample_project)])

        assert result.exit_code == 0
        assert "Found" in result.output or "results" in result.output