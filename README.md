# Kozzle Word Grouper

A Python CLI tool for organizing Korean words using LLM-based categorization or embedding-based clustering.

## Features

- **LLM-Based Categorization**: Categorize Korean words using Ollama (exaone3.5:7.8b) with three classification types:
  - 하위개념 (Hyponym/Concept Hierarchy)
  - 기능 (Function/Role)
  - 사용맥락 (Usage Context)
- **Embedding-Based Clustering**: Alternative approach using semantic embeddings + HDBSCAN clustering
- **Supabase Integration**: Fetch Korean words directly from Supabase with connection pooling and retry logic
- **Word Caching**: Cache fetched words locally to avoid repeated Supabase API calls
- **Resume Capability**: Resume categorization from where you left off
- **Database-Ready Output**: Export results in JSON format optimized for Supabase/PostgreSQL import
- **Korean Language Support**: Specialized for Korean word analysis with exaone3.5:7.8b model

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/kozzle-word-grouper.git
cd kozzle-word-grouper

# Install dependencies using uv
uv sync

# Install Ollama (required for LLM-based categorization)
# See: https://ollama.ai/
curl -fsSL https://ollama.com/install.sh | sh

# Pull the Korean language model
ollama pull exaone3.5:7.8b
```

## Setup

1. Copy the example environment file:

```bash
cp .env.example .env
```

2. Add your Supabase credentials to `.env`:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-or-service-role-key

# Optional: Override Ollama host (default: http://localhost:11434)
OLLAMA_HOST=http://localhost:11434
```

3. Ensure your Supabase table has Korean words:
   - Table name: `kor_word` (configurable)
   - Required columns: `public_id`, `lemma`, `definition` (optional), `level` (optional for filtering)

## Usage

### LLM-Based Categorization (Recommended)

Categorize Korean words using three classification types:

```bash
# Basic usage
uv run kozzle-word-grouper categorize

# With filtering (level 1 or 2, lemma length >= 2)
uv run kozzle-word-grouper categorize \
  --filter-level 1 --filter-level 2 \
  --min-lemma-length 2

# Test with subset
uv run kozzle-word-grouper categorize --subset 100

# Resume from categorization cache
uv run kozzle-word-grouper categorize --resume

# Use previously cached words (skip Supabase fetch)
uv run kozzle-word-grouper categorize --from-cache

# Custom cache file location
uv run kozzle-word-grouper categorize \
  --from-cache \
  --cache-file /path/to/words_cache.json
```

### Embedding-Based Clustering (Alternative)

Group words using semantic embeddings and HDBSCAN:

```bash
# Basic usage
uv run kozzle-word-grouper group

# With filtering
uv run kozzle-word-grouper group \
  --filter-level 1 --filter-level 2 \
  --min-lemma-length 2

# Adjust cluster size
uv run kozzle-word-grouper group --min-cluster-size 15
```

### View Information

```bash
# Display connection pool stats
uv run kozzle-word-grouper pool-info
```

## Output Files

### LLM Categorization Output

**`word_categorization.json`** - Database-ready JSON:

```json
{
  "version": "1.0",
  "metadata": {
    "total_words": 5874,
    "processed_words": 5874,
    "processed_at": "2026-04-02T21:30:41Z",
    "model": "exaone3.5:7.8b",
    "classification_types": ["하위개념", "기능", "사용맥락"],
    "schema_version": "1.0"
  },
  "categorizations": [
    {
      "public_id": "550e8400-...",
      "lemma": "개",
      "definition": "사람이 집에서 기르는 동물",
      "categories": {
        "하위개념": ["동물", "생물", "포유류"],
        "기능": ["애완동물", "반려동물"],
        "사용맥락": ["일상대화", "반려동물관련"]
      },
      "processed_at": "2026-04-02T21:30:41Z",
      "model_version": "exaone3.5:7.8b"
    }
  ],
  "category_index": {
    "하위개념": {
      "동물": [{"public_id": "...", "lemma": "개"}, ...]
    },
    "기능": {...},
    "사용맥락": {...}
  },
  "statistics": {
    "하위개념": {
      "total_categories": 50,
      "avg_words_per_category": 117.5
    }
  }
}
```

**`categorization_cache.json`** - Categorization progress for resume

**`words_cache.json`** - Cached Korean words from Supabase

### Embedding Clustering Output

**`word_groups.json`** - Clusters with Korean labels

**`word_groups.csv`** - Flat CSV with word-cluster mappings

**`cluster_summary.txt`** - Human-readable summary

**`label_cache.json`** - Cached cluster labels

## Database Integration

### PostgreSQL/Supabase Schema

```sql
-- Main categorizations table (one row per word)
CREATE TABLE word_categorizations (
  id SERIAL PRIMARY KEY,
  public_id UUID REFERENCES kor_word(public_id) UNIQUE NOT NULL,
  lemma TEXT NOT NULL,
  definition TEXT,
  categories JSONB NOT NULL,
  processed_at TIMESTAMP NOT NULL,
  model_version TEXT NOT NULL DEFAULT 'exaone3.5:7.8b',
  created_at TIMESTAMP DEFAULT NOW()
);

-- Category lookup index (normalized for fast queries)
CREATE TABLE category_index (
  id SERIAL PRIMARY KEY,
  classification_type TEXT NOT NULL,
  category_name TEXT NOT NULL,
  public_id UUID REFERENCES kor_word(public_id) NOT NULL,
  lemma TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_category_lookup ON category_index(classification_type, category_name);
CREATE INDEX idx_categories_gin ON word_categorizations USING GIN(categories);
```

### Import Script

```python
import json
from supabase import Client

def import_categorizations(supabase: Client, json_file: str):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Import to word_categorizations
    for record in data['categorizations']:
        supabase.table('word_categorizations').insert({
            'public_id': record['public_id'],
            'lemma': record['lemma'],
            'definition': record['definition'],
            'categories': record['categories'],
            'processed_at': record['processed_at']
        }).execute()
    
    # Populate category_index
    for class_type, categories in data['category_index'].items():
        for category_name, words in categories.items():
            for word in words:
                supabase.table('category_index').insert({
                    'classification_type': class_type,
                    'category_name': category_name,
                    'public_id': word['public_id'],
                    'lemma': word['lemma']
                }).execute()
```

## Performance

### LLM Categorization

- **Processing Rate**: ~8-10 words/second (with 4 concurrent workers)
- **Total Time** (5874 words): ~10-12 minutes
- **Network Calls**: 3 LLM calls per word (하위개념, 기능, 사용맥락)
- **Resume**: Can resume from cache if interrupted

### Embedding Clustering

- **Embedding Generation**: ~2 words/second (Ollama exaone3.5:7.8b)
- **Total Time** (5874 words): ~12 minutes
- **Memory**: ~500MB for embeddings

## Word Caching System

The tool caches fetched Korean words to avoid repeated Supabase API calls:

### First Run (fetch and cache)

```bash
uv run kozzle-word-grouper categorize \
  --filter-level 1 --filter-level 2 \
  --min-lemma-length 2
```

Output:
```
✓ Fetched 5874 words from Supabase
✓ Cached to output/words_cache.json (863 KB)
✓ Categorizing words...
```

### Subsequent Runs (use cached words)

```bash
uv run kozzle-word-grouper categorize --from-cache
```

Output:
```
✓ Loaded 5874 words from cache (cached at: 2026-04-02T21:37:18)
✓ Skipped Supabase fetch
✓ Categorizing words...
```

### Benefits

1. **Faster Iteration**: Re-run categorization without re-fetching
2. **Offline Work**: Work without Supabase connection after initial fetch
3. **Cost Savings**: Reduce Supabase API calls
4. **Resume**: Continue categorization from where you left off

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Input (Supabase)                      │
│          KoreanWord {public_id, lemma, definition}       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                 Word Cache (words_cache.json)            │
│   - First run: Fetch from Supabase and cache            │
│   - Subsequent: Load from cache (--from-cache)           │
└────────────────────┬────────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
┌──────────────────┐   ┌──────────────────┐
│  LLM Categorize  │   │  Embed + Cluster │
│  (categorize)     │   │  (group)         │
│                   │   │                   │
│  3 prompts/word:  │   │  - Ollama embed   │
│  - 하위개념        │   │  - HDBSCAN        │
│  - 기능           │   │  - Label clusters │
│  - 사용맥락        │   │                   │
└──────────┬────────┘   └──────────┬────────┘
           │                        │
           └────────┬───────────────┘
                    ▼
┌─────────────────────────────────────────────────────────┐
│                    Output (JSON)                         │
│  - Database-ready format                                 │
│  - Korean labels                                         │
│  - Category index for fast lookups                       │
│  - Statistics                                            │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
kozzle-word-grouper/
├── src/kozzle_word_grouper/
│   ├── __init__.py              # Package init
│   ├── __main__.py              # Entry point
│   ├── cli.py                   # CLI commands (categorize, group)
│   ├── core.py                  # Main pipeline orchestrator
│   ├── supabase_client.py       # Supabase integration + connection pool
│   ├── categorizer.py           # LLM categorization logic
│   ├── category_aggregator.py   # Category aggregation and stats
│   ├── embeddings.py            # Ollama embedding generation
│   ├── clustering.py            # HDBSCAN clustering
│   ├── labeler.py                # Korean cluster label generation
│   ├── export.py                 # Export to JSON/CSV/text
│   ├── models.py                 # KoreanWord dataclass
│   ├── exceptions.py             # Custom exceptions
│   ├── connection_pool.py        # HTTP connection pool manager
│   ├── retry.py                   # Retry logic with backoff
│   └── utils.py                   # Utilities
├── tests/
│   ├── conftest.py              # Pytest fixtures
│   └── test_*.py                # Test modules
├── database_schema.sql          # Database schema for Supabase
├── pyproject.toml               # Dependencies and config
├── .env.example                 # Environment variables template
├── AGENTS.md                    # Development guidelines
└── README.md                    # This file
```

## Configuration

### Environment Variables

```env
# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-or-service-role-key

# Ollama
OLLAMA_HOST=http://localhost:11434

# Connection Pool
SUPABASE_MAX_CONNECTIONS=10
SUPABASE_IDLE_TIMEOUT=0.1
SUPABASE_MAX_RETRIES=3
```

### CLI Options

**`categorize` command:**

```bash
uv run kozzle-word-grouper categorize [OPTIONS]

Options:
  -t, --table TEXT               Name of the table containing Korean words
  -l, --lemma-column TEXT        Name of the lemma column
  -d, --definition-column TEXT   Name of the definition column
  -p, --public-id-column TEXT    Name of the public ID column
  --level-column TEXT            Name of the level column for filtering
  -f, --filter-level INTEGER     Filter by level (e.g., -f 1 -f 2)
  -m, --min-lemma-length INTEGER Minimum lemma length
  --model TEXT                   Ollama model for categorization
  -o, --output-dir TEXT          Directory to save output files
  --ollama-host TEXT             Ollama server URL
  --subset INTEGER               Number of words to process (for testing)
  --resume                       Resume from categorization cache
  --from-cache                   Load words from cache instead of Supabase
  --cache-file PATH              Path to cache file
```

**`group` command:**

```bash
uv run kozzle-word-grouper group [OPTIONS]

Options:
  # Same as categorize, plus:
  -c, --min-cluster-size INTEGER  Minimum words per cluster
  --output-format [json|csv|summary]  Output formats
```

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=term-missing

# Run specific test
uv run pytest tests/test_categorizer.py::test_categorize_word

# Type checking
uv run mypy src
```

## Development

### Code Style

```bash
# Format code
uv run ruff format .

# Check linting
uv run ruff check .

# Type checking
uv run mypy src
```

### Running Linters

This project enforces code quality checks:

```bash
# Run all quality checks
uv run ruff format .
uv run ruff check .
uv run mypy src
uv run pytest
```

## Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve
```

### Supabase Connection Issues

- Verify `SUPABASE_URL` and `SUPABASE_KEY` in `.env`
- Check your Supabase project settings
- Ensure your key has appropriate permissions
- Check connection pool settings

### Memory Issues

- Use `--subset` to process smaller batches
- Use `--from-cache` to avoid re-fetching words
- Use `--resume` to continue categorization from checkpoint

### Categorization Failures

- Check categorization cache at `output/categorization_cache.json`
- Resume with `--resume` flag
- Check Ollama logs for errors

## Performance Tips

1. **Use Word Cache**: Always use `--from-cache` for subsequent runs
2. **Resume Categorization**: Use `--resume` to continue from checkpoint
3. **Test with Subset**: Use `--subset 100` to test before full run
4. **Monitor Ollama**: Use `ollama logs` to monitor LLM performance
5. **Concurrent Workers**: Default is 4 workers, adjust in `core.py` if needed

## Examples

### Example 1: Full Production Run

```bash
# First run: fetch and cache words
uv run kozzle-word-grouper categorize \
  --filter-level 1 --filter-level 2 \
  --min-lemma-length 2 \
  --output-dir output

# Takes ~10-12 minutes for 5874 words
# Creates:
# - output/words_cache.json (cached words)
# - output/categorization_cache.json (categorization progress)
# - output/word_categorization.json (final results)
```

### Example 2: Resume Interrupted Run

```bash
# If categorization is interrupted, just run again with --resume
uv run kozzle-word-grouper categorize \
  --from-cache \
  --resume \
  --output-dir output

# Resumes from checkpoint
# Loads words from cache
# Continues categorization from where it left off
```

### Example 3: Test with Subset

```bash
# Test with 100 words to verify configuration
uv run kozzle-word-grouper categorize \
  --subset 100 \
  --filter-level 1 --filter-level 2 \
  --min-lemma-length 2 \
  --output-dir output/test

# Quick validation before full run
```

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments

- [Ollama](https://ollama.ai/) for local LLM inference
- [Exaone3.5](https://huggingface.co/LGAI-EXAONE/exaone-3.5-7.8b-instruct) for Korean language model
- [Supabase](https://supabase.com/) for database infrastructure
- [HDBSCAN](https://hdbscan.readthedocs.io/) for clustering algorithm