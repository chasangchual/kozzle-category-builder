# AGENTS.md - Development Guidelines for kozzle-word-grouper

## Architecture Overview

### Core Components
- **Supabase Client** (`supabase_client.py`) - Fetches Korean words with connection pooling (30s timeout), retry logic, and database-level filtering
- **Embedding Generator** (`embeddings.py`) - Generates embeddings using Ollama API (exaone3.5:7.8b model)
- **Cluster Labeler** (`labeler.py`) - Generates Korean category names for clusters using Ollama
- **Categorizer** (`categorizer.py`) - LLM-based word categorization with rate limiting (2 calls/sec), batch processing (20 words/batch), and efficient caching (every 100 words)
- **Category Compressor** (`category_compressor.py`) - Merges and normalizes categories
- **Category Aggregator** (`category_aggregator.py`) - Aggregates categorization results
- **Predefined Categorizer** (`predefined_categorizer.py`) - Binary classification into predefined categories (5 words/batch, 2s delay)
- **Word Clusterer** (`clustering.py`) - HDBSCAN-based clustering
- **Word Grouper Pipeline** (`core.py`) - Main orchestration class
- **Models** (`models.py`) - Data classes (KoreanWord, ClusteredWord, WordCluster)

### Data Flow
```
Supabase (kor_word table)
↓ (public_id, lemma, definition, level)
SupabaseClient.fetch_korean_words()
↓
[KoreanWord(public_id, lemma, definition)]
↓
Option 1: Clustering Pipeline
  → EmbeddingGenerator → HDBSCAN → ClusterLabeler → Export
Option 2: Categorization Pipeline  
  → Categorizer (LLM) → CategoryAggregator → Export
Option 3: Predefined Classification
  → PredefinedCategorizer → Export
```

## Build/Lint/Test Commands

### Setup
```bash
uv sync
```

### Running the CLI
```bash
uv run kozzle-word-grouper --help
uv run kozzle-word-grouper categorize --filter-level 1 --subset 100
uv run kozzle-word-grouper group --min-cluster-size 10
uv run kozzle-word-grouper compress --input-file output/word_categorization.json
uv run kozzle-word-grouper classify --categories-file kor_words_catetoris.json
```

### Testing

Run all tests:
```bash
uv run pytest
```

Run a single test file:
```bash
uv run pytest tests/test_core.py
```

Run a single test function:
```bash
uv run pytest tests/test_core.py::test_pipeline_initialization
```

Run tests with verbose output:
```bash
uv run pytest -v
```

Run tests with coverage:
```bash
uv run pytest --cov=src --cov-report=term-missing
```

Run specific test markers (if defined):
```bash
uv run pytest -m marker_name
```

### Linting and Type Checking

Run linting:
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

## Code Style Guidelines

### Project Structure
```
kozzle-word-grouper/
├── src/kozzle_word_grouper/
│   ├── __init__.py              # Package init with version and exports
│   ├── __main__.py              # CLI entry point
│   ├── cli.py                  # Click-based CLI commands
│   ├── core.py                 # Main pipeline orchestration
│   ├── models.py               # Data classes (KoreanWord, ClusteredWord)
│   ├── exceptions.py           # Custom exception hierarchy
│   ├── utils.py                # Utilities (logging, batching)
│   ├── supabase_client.py      # Supabase integration
│   ├── embeddings.py           # Embedding generation
│   ├── clustering.py           # HDBSCAN clustering
│   ├── labeler.py              # Cluster labeling
│   ├── categorizer.py           # LLM categorization
│   ├── category_aggregator.py  # Category aggregation
│   ├── category_compressor.py  # Category merging
│   ├── predefined_categorizer.py # Binary classification
│   ├── export.py               # Export functionality
│   ├── connection_pool.py      # HTTP connection pool
│   ├── retry.py                # Retry logic
│   └── monitoring.py           # Logging/monitoring
├── tests/
│   ├── conftest.py             # Pytest fixtures
│   └── test_*.py               # Test modules
├── pyproject.toml              # Project config
├── mypy.ini                    # MyPy configuration
└── .env.example                # Environment template
```

### Imports
- Use absolute imports from package root
- Group in order: standard library, third-party, local application
- Separate each group with a blank line
- Example:
```python
import json
from pathlib import Path
from typing import Any

import click
import numpy as np
from numpy.typing import NDArray

from kozzle_word_grouper.core import WordGrouperPipeline
from kozzle_word_grouper.exceptions import WordGrouperError
from kozzle_word_grouper.models import KoreanWord
from kozzle_word_grouper.utils import get_logger
```

### Formatting
- Use `ruff format` (Black-compatible, 88 character line length)
- Use double quotes for strings consistently
- Use 4 spaces for indentation (no tabs)
- Add trailing comma in multi-line collections
- Example:
```python
result = pipeline.run(
    table_name="kor_word",
    filter_level=[1, 2],
    min_lemma_length=2,
    show_progress=True,
)
```

### Type Annotations
- Use type hints for ALL function signatures (enforced by mypy)
- Use Python 3.10+ syntax: `str | None` instead of `Optional[str]`
- Use `list[str]` instead of `List[str]`, `dict[str, Any]` instead of `Dict[str, Any]`
- For NumPy arrays, use: `NDArray[np.floating[Any]]` or `NDArray[np.integer[Any]]`
- Always add return type hints: `-> None` or `-> dict[str, Any]`
- Example:
```python
from typing import Any
from numpy.typing import NDArray

def fetch_korean_words(
    self,
    table_name: str = "kor_word",
    filter_level: list[int] | None = None,
    min_lemma_length: int | None = None,
) -> list[KoreanWord]:
    ...
```

### Data Classes
- Use `@dataclass` for model classes
- Example:
```python
from dataclasses import dataclass

@dataclass
class KoreanWord:
    public_id: str
    lemma: str
    definition: str | None
    
    def get_text_for_embedding(self) -> str:
        if self.definition and self.definition.strip():
            return self.definition
        return self.lemma
```

### Naming Conventions
- **Functions/Methods**: `snake_case` (e.g., `fetch_korean_words`, `generate_embeddings`)
- **Classes**: `PascalCase` (e.g., `WordGrouperPipeline`, `CategoryCompressor`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_RETRIES`, `CLASSIFICATION_TYPES`)
- **Private methods**: Prefix with underscore (e.g., `_validate_connection`, `_build_prompt`)
- **Type Aliases**: `PascalCase` (e.g., `WordList = list[str]`)
- **CLI commands**: Use hyphens (e.g., `--filter-level`, `--min-lemma-length`)

### Error Handling
- Raise specific exceptions from `exceptions.py`:
  - `WordGrouperError` - base exception
  - `SupabaseConnectionError`, `DataRetrievalError` - Supabase errors
  - `OllamaConnectionError`, `OllamaModelError` - LLM errors
  - `CategorizationError`, `ClusteringError` - processing errors
- Use exception chaining with `from e`:
```python
from kozzle_word_grouper.exceptions import WordGrouperError

try:
    response = requests.get(url, timeout=5)
except requests.RequestException as e:
    raise OllamaConnectionError(f"Failed to connect: {e}") from e
```
- Use try/except for expected failures, let unexpected errors propagate
- Use logging module (`get_logger(__name__)`) for non-CLI output:
```python
from kozzle_word_grouper.utils import get_logger

logger = get_logger(__name__)

def run_categorization(self) -> dict[str, Any]:
    logger.info("Starting categorization pipeline")
    ...
    logger.error(f"Categorization failed: {e}")
```

### CLI Design
- Use `click` for CLI argument parsing
- Use sensible defaults with clear help text
- Exit with exit code 1 on errors: `raise SystemExit(1)`
- Example:
```python
@click.command()
@click.option("--filter-level", "-f", multiple=True, type=int,
              help="Filter by level (e.g., -f 1 -f 2 for level 1 or 2)")
@click.option("--subset", type=int, help="Number of words to process")
def categorize(filter_level: tuple[int, ...], subset: int | None) -> None:
    """Categorize Korean words using LLM classification."""
    try:
        pipeline = WordGrouperPipeline()
        pipeline.run_categorization(filter_level=list(filter_level))
    except WordGrouperError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)
```

### Testing Conventions
- Place tests in `tests/` directory
- Name files as `test_<module_name>.py`
- Name functions as `test_<function_name>_<scenario>`
- Use pytest fixtures from `conftest.py`
- Add return type hints: `def test_something() -> None:`
- Use `@patch` for mocking, `MagicMock` and `Mock` from `unittest.mock`
- Use `pytest.raises()` for exception testing
- Use `tmp_path` fixture for temporary directories
- Example:
```python
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

def test_pipeline_with_no_words(temp_output_dir: Path) -> None:
    """Test pipeline handles empty word list."""
    with patch("kozzle_word_grouper.core.SupabaseClient") as mock_supabase_class:
        mock_supabase = MagicMock()
        mock_supabase.fetch_korean_words.return_value = []
        mock_supabase_class.return_value = mock_supabase
        
        pipeline = WordGrouperPipeline(output_dir=temp_output_dir)
        
        with pytest.raises(WordGrouperError, match="No words found"):
            pipeline.run()
```

### Docstrings
- Use Google-style docstrings for all public modules, classes, and functions
- Include Args, Returns, and Raises sections
- Example:
```python
def run_categorization(
    self,
    table_name: str = "kor_word",
    filter_level: list[int] | None = None,
    subset: int | None = None,
) -> dict[str, Any]:
    """Run LLM-based categorization pipeline.
    
    Args:
        table_name: Name of the Supabase table.
        filter_level: List of levels to include (e.g., [1, 2]).
        subset: Number of words to process (None for all).
        
    Returns:
        Dictionary with categorizations and statistics.
        
    Raises:
        WordGrouperError: If categorization fails.
    """
```

## Important Notes

- Always use `uv run` to execute commands in the virtual environment
- Add dependencies via `uv add`, not `pip install`
- Run linter and type checker before committing: `uv run ruff check . && uv run mypy src`
- All functions must have type hints and docstrings
- Keep functions small and focused (single responsibility)
- Use `Path` from pathlib for file paths: `Path(output_dir)`
- For long-running operations, use `show_progress: bool` parameter
- Cache results when possible (use `--resume` flag for caching)
- Connection pool is managed automatically via `connection_pool.py`

## Performance Considerations

### Connection Pooling
- Connection pool uses 30-second idle timeout (configurable via `SUPABASE_IDLE_TIMEOUT`)
- Designed to keep connections alive during long LLM operations (2-10 seconds)
- Prevents connection thrashing and SSL handshake overhead
- Do NOT reduce idle timeout below 30 seconds for production use

### Rate Limiting
- LLM API calls are rate-limited to 2.0 calls/second by default
- Prevents overwhelming Ollama with concurrent requests
- Rate limiter class in `categorizer.py` and `predefined_categorizer.py`
- Can be configured via `rate_limit` parameter in Categorizer init

### Batch Processing
- Categorizer processes 20 words per batch with 0.5s delay between batches
- PredefinedCategorizer processes 5 words per batch with 2.0s delay (150 calls/word)
- Prevents burst load on LLM server
- Do NOT increase batch size or decrease batch delay without testing

### Cache Management
- Cache saves every 100 words for Categorizer (configurable via `cache_save_interval`)
- Cache saves every 50 words for PredefinedCategorizer
- Progressive slowdown caused by growing dictionary serialization
- Do NOT reduce cache save interval for large datasets

### Database-Level Filtering
- Lemma length filtering happens at PostgreSQL level via `length()` function
- Reduces network bandwidth and memory usage
- Always prefer database-level filtering over client-side filtering
- Use `.gte(f"length({column})", value)` for efficient filtering

### Performance Testing Commands
```bash
# Test small dataset performance
uv run kozzle-word-grouper categorize --subset 100 --verbose

# Monitor for:
# - Consistent processing rate (words/second should not decline)
# - No "Connection error" or "Timeout" messages
# - Steady state execution (no progressive slowdown)

# Test predefined categorization (most intensive)
uv run kozzle-word-grouper classify --subset 10 --verbose
```

### Common Performance Issues

**Issue**: Progressive slowdown during execution
- **Cause**: Connection pool idle timeout too low (100ms default was too aggressive)
- **Fix**: Changed to 30s timeout, connections stay alive during LLM calls

**Issue**: Growing cache save time
- **Cause**: JSON serialization of entire dictionary on every save
- **Fix**: Reduced save frequency (every 100 words instead of 50)

**Issue**: Ollama timeout errors
- **Cause**: Unlimited concurrent requests overwhelming LLM
- **Fix**: Added rate limiting (2 calls/sec) and batch processing delays

**Issue**: Slow database queries
- **Cause**: Client-side filtering of fetched data
- **Fix**: Moved filtering to PostgreSQL level

### Expected Performance Metrics

| Operation | Rate | Notes |
|-----------|------|-------|
| Word categorization | ~0.5-1.0 words/sec | 3 LLM calls per word |
| Predefined categorization | ~0.05 words/sec | 150 LLM calls per word |
| Database fetch | ~1000 words/batch | With filtering at DB level |
| Cache save | Every 100 words | JSON serialization |
| Connection reuse | 30+ seconds | Stay alive during LLM calls |