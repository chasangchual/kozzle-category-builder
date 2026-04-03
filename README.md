# Kozzle Word Grouper

A Python CLI tool that fetches words from Supabase and groups them by semantic meaning using local embedding models.

## Features

- **Supabase Integration**: Fetch words directly from your Supabase database
- **Semantic Clustering**: Group words by meaning using sentence embeddings
- **Local Models**: Use HuggingFace sentence-transformers (no API calls required)
- **Auto-Detection**: HDBSCAN automatically determines optimal number of clusters
- **Multiple Exports**: Export results as JSON, CSV, or text summary

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/kozzle-word-grouper.git
cd kozzle-word-grouper

# Install dependencies using uv
uv sync
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
```

3. Ensure your Supabase table has the words you want to cluster. Default expected table:
   - Table name: `words`
   - Column name: `word`

## Usage

### Basic Usage

```bash
# Run with default settings
uv run kozzle-word-grouper group

# This will:
# - Fetch words from Supabase (table: 'words', column: 'word')
# - Generate embeddings using 'all-mpnet-base-v2' model
# - Cluster words using HDBSCAN (auto-detect number of clusters)
# - Export results to ./output/
```

### Advanced Options

```bash
# Specify custom table and column
uv run kozzle-word-grouper group \
  --table my_words \
  --word-column text

# Use a different embedding model
uv run kozzle-word-grouper group \
  --model all-MiniLM-L6-v2 \
  --min-cluster-size 10

# Specify output directory and formats
uv run kozzle-word-grouper group \
  --output-dir ./results \
  --output-format json csv

# Cache embeddings for reuse
uv run kozzle-word-grouper group \
  --save-embeddings ./embeddings.npy

# Use cached embeddings
uv run kozzle-word-grouper group \
  --load-embeddings ./embeddings.npy
```

### View Information

```bash
# Display information about available models
uv run kozzle-word-grouper info
```

## Output Files

The tool generates three output files:

1. **`word_groups.json`**: Structured JSON with cluster information
2. **`word_groups.csv`**: Flat CSV with word-cluster mappings
3. **`cluster_summary.txt`**: Human-readable summary

### JSON Output Format

```json
{
  "cluster_0": {
    "cluster_id": 0,
    "label": "cluster_0",
    "word_count": 152,
    "representative_words": ["dog", "cat", "bird"],
    "words": ["dog", "cat", "bird", ...]
  },
  "noise": {
    "cluster_id": -1,
    "label": "noise",
    "word_count": 5,
    "words": [...]
  }
}
```

### CSV Output Format

```csv
word,cluster_id,cluster_label
dog,0,cluster_0
cat,0,cluster_0
car,1,cluster_1
```

## Supported Models

- **`all-MiniLM-L6-v2`**: Fast, 384 dimensions (recommended for large datasets)
- **`all-mpnet-base-v2`**: Balanced, 768 dimensions (default, best quality)
- **`paraphrase-multilingual-mpnet-base-v2`**: Multilingual support
- **`all-roberta-large-v1`**: High quality, 1024 dimensions

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_clustering.py

# Run specific test
uv run pytest tests/test_clustering.py::test_fit_predict_small_dataset
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

### Project Structure

```
kozzle-word-grouper/
├── src/kozzle_word_grouper/
│   ├── __init__.py
│   ├── __main__.py      # Entry point
│   ├── cli.py           # CLI interface
│   ├── core.py          # Main pipeline
│   ├── supabase_client.py  # Supabase integration
│   ├── embeddings.py    # Embedding generation
│   ├── clustering.py    # Clustering logic
│   ├── export.py        # Export functionality
│   ├── exceptions.py    # Custom exceptions
│   └── utils.py         # Utilities
├── tests/
│   ├── conftest.py      # Pytest fixtures
│   └── test_*.py        # Test modules
├── pyproject.toml
└── README.md
```

## How It Works

1. **Data Retrieval**: Fetches words from your Supabase database
2. **Embedding Generation**: Converts each word to a dense vector using sentence-transformers
3. **Clustering**: Uses HDBSCAN to cluster words based on cosine similarity
4. **Export**: Saves results with cluster assignments and representative words

## Performance

For large datasets (10k-100k words):

- Use `all-MiniLM-L6-v2` for faster processing
- Adjust `--batch-size` for embedding generation
- Save embeddings with `--save-embeddings` to avoid regenerating

## Troubleshooting

### Module Not Found

```bash
# Ensure you're using uv run
uv run kozzle-word-grouper --help
```

### Supabase Connection Issues

- Verify `SUPABASE_URL` and `SUPABASE_KEY` in `.env`
- Check your Supabase project settings
- Ensure your key has appropriate permissions

### Memory Issues with Large Datasets

- Use `--load-embeddings` and `--save-embeddings` to process in stages
- Start with a smaller model like `all-MiniLM-L6-v2`

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.