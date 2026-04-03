# AGENTS.md - Development Guidelines for kozzle-word-grouper

## Architecture

### Core Components
- **Supabase Client** (`supabase_client.py`) - Fetches Korean words from `kor_word` table with connection pooling and retry logic
- **Embedding Generator** (`embeddings.py`) - Generates embeddings using Ollama API (exaone3.5:7.8b model)
- **Cluster Labeler** (`labeler.py`) - Generates Korean category names for clusters using Ollama
- **Word Clusterer** (`clustering.py`) - Clusters words using HDBSCAN
- **Connection Pool Manager** (`connection_pool.py`) - Manages HTTP connection pool for Supabase API calls
- **Retry Logic** (`retry.py`) - Automatic retry on transient failures with exponential backoff

### Data Flow
```
Supabase (kor_word table)
↓ (public_id, lemma, definition)
SupabaseClient.fetch_korean_words()
↓
[{"public_id": "...", "lemma": "...", "definition": "..."}]
↓
EmbeddingGenerator.generate_embeddings()
↓ (definition or lemma fallback)
Ollama API /api/embeddings (exaone3.5:7.8b)
↓
numpy array of embeddings
↓
WordClusterer.fit_predict()
↓
cluster labels
↓
ClusterLabeler.label_clusters()
↓ (generate Korean category names)
{"동물": [...], "색상": [...]}
↓
WordGroupExporter.export_all()
↓
JSON/CSV/Summary files with Korean labels

## Build/Lint/Test Commands

### Setup
```bash
uv sync
```

### Running the CLI
```bash
uv run kozzle-word-grouper [args]
```

### Testing

Run all tests:
```bash
uv run pytest
```

Run a single test file:
```bash
uv run pytest tests/test_module.py
```

Run a single test function:
```bash
uv run pytest tests/test_module.py::test_function_name
```

Run tests with verbose output:
```bash
uv run pytest -v
```

Run tests with coverage:
```bash
uv run pytest --cov=src --cov-report=term-missing
```

Run specific test markers:
```bash
uv run pytest -m marker_name
```

### Linting and Type Checking

Run all linting (if configured):
```bash
uv run ruff check .
```

Run type checking:
```bash
uv run mypy src
```

Format code:
```bash
uv run ruff format .
```

Check formatting without applying:
```bash
uv run ruff format --check .
```

### Dependency Management

Add a dependency:
```bash
uv add package-name
```

Add a development dependency:
```bash
uv add --dev package-name
```

Update dependencies:
```bash
uv lock --upgrade
```

## Code Style Guidelines

### Project Structure
```
kozzle-word-grouper/
├── src/
│   └── kozzle_word_grouper/
│       ├── __init__.py          # Package init with version
│       ├── __main__.py          # CLI entry point
│       ├── cli.py               # CLI argument parsing (click)
│       ├── core.py              # Main pipeline orchestration
│       ├── supabase_client.py   # Supabase integration
│       ├── embeddings.py        # Embedding generation
│       ├── clustering.py        # HDBSCAN clustering logic
│       ├── export.py            # Export functionality
│       ├── exceptions.py        # Custom exception hierarchy
│       └── utils.py             # Utility functions
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Pytest fixtures
│   └── test_*.py                # Test modules
├── .env.example                 # Environment variables template
├── mypy.ini                     # MyPy configuration
├── pyproject.toml               # Project config (uv, ruff, pytest)
└── README.md
```

### Imports
- Use absolute imports from the package root
- Group imports in this order:
  1. Standard library
  2. Third-party libraries
  3. Local application imports
- Separate each group with a blank line
- Example:
```python
import os
import sys
from pathlib import Path

import click
import pytest

from kozzle_word_grouper.core import process_word
from kozzle_word_grouper.utils import helper_function
```

### Formatting
- Use `ruff format` for code formatting (or Black-compatible)
- Maximum line length: 88 characters
- Use double quotes for strings (or be consistent)
- Use 4 spaces for indentation (no tabs)
- Add trailing comma in multi-line collections

### Type Annotations
- Use type hints for ALL function signatures (enforced by mypy)
- Use Python 3.10+ syntax for unions: `str | None` instead of `Optional[str]`
- Use modern collection types: `list[str]` instead of `List[str]`
- Use specific types: `dict[int, dict[str, Any]]` instead of generic dict
- For NumPy arrays, use: `NDArray[np.floating[Any]]` or `NDArray[np.integer[Any]]`
- Example:
```python
from typing import Any
from numpy.typing import NDArray
import numpy as np

def process_words(
    words: list[str],
    embeddings: NDArray[np.floating[Any]],
) -> dict[int, dict[str, Any]]:
    ...
```

### Naming Conventions
- **Functions/Methods**: snake_case (e.g., `process_word_group`)
- **Classes**: PascalCase (e.g., `WordGrouper`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_WORD_LENGTH`)
- **Private methods**: Prefix with underscore (e.g., `_internal_process`)
- **Type Aliases**: PascalCase (e.g., `WordList = list[str]`)
- **CLI commands**: Use hyphens in CLI args (e.g., `--group-size`)

### Error Handling
- Raise specific exceptions with descriptive messages
- Use custom exception hierarchy (see `exceptions.py`):
  - `WordGrouperError` - base exception
  - `SupabaseConnectionError`, `DataRetrievalError`, `EmbeddingError`, `ClusteringError`, `ExportError` - specific exceptions
- Use exception chaining with `from e`:
```python
from kozzle_word_grouper.exceptions import WordGrouperError

try:
    # Some operation
    pass
except SomeError as e:
    raise WordGrouperError(f"Failed to process: {e}") from e
```

- Use try/except blocks for expected failure cases
- Let unexpected errors propagate with full traceback
- Use logging instead of print statements for non-CLI output:
```python
from kozzle_word_grouper.utils import get_logger

logger = get_logger(__name__)

def process_words(words: list[str]) -> list[str]:
    logger.debug(f"Processing {len(words)} words")
    ...
```

### CLI Design
- Use `click` or `typer` for CLI argument parsing
- Provide helpful docstrings for commands
- Use sensible defaults with clear help text
- Exit with appropriate exit codes (0 for success, non-zero for errors)

### Documentation
- Use docstrings for all public modules, classes, and functions
- Follow Google or NumPy docstring style:
```python
def process_words(words: list[str], group_size: int = 5) -> list[str]:
    """Process a list of words and group them.
    
    Args:
        words: List of words to process.
        group_size: Number of words per group.
        
    Returns:
        List of processed word groups.
        
    Raises:
        WordGrouperError: If processing fails.
    """
    ...
```

### Testing Conventions
- Place tests in the `tests/` directory
- Name test files as `test_<module_name>.py`
- Name test functions as `test_<function_name>_<scenario>`
- Use `pytest` fixtures for common setup (defined in `conftest.py`)
- Add return type hints to test functions: `def test_something() -> None:`
- Use `@patch` for mocking external dependencies
- Use `pytest.raises()` for testing exceptions
- Use `tmp_path` fixture from pytest for temporary directories
- Use `MagicMock` and `Mock` from `unittest.mock` for mocking
- Example:
```python
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
import pytest
import numpy as np
from numpy.typing import NDArray
from kozzle_word_grouper.core import WordGrouperPipeline
from kozzle_word_grouper.exceptions import WordGrouperError

def test_pipeline_initialization() -> None:
    """Test pipeline initialization."""
    pipeline = WordGrouperPipeline(model_name="test-model")
    assert pipeline.model_name == "test-model"

def test_pipeline_run_no_words(temp_output_dir: Path) -> None:
    """Test pipeline with no words."""
    with patch("kozzle_word_grouper.core.SupabaseClient") as mock_supabase_class:
        mock_supabase = MagicMock()
        mock_supabase.fetch_words.return_value = []
        mock_supabase_class.return_value = mock_supabase
        
        pipeline = WordGrouperPipeline(output_dir=temp_output_dir)
        
        with pytest.raises(WordGrouperError):
            pipeline.run()
```

## Important Notes
- Always use `uv run` to execute commands in the virtual environment
- Dependencies should be added via `uv add`, not `pip install`
- Run linting and type checking before committing changes
- All functions should have type hints and docstrings
- Keep functions small and focused on a single responsibility