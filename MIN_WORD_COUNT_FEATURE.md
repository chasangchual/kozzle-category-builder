# Category Filtering Feature (--min-word-count)

## Overview

Added ability to filter categories by minimum word count, keeping only categories with sufficient words to be meaningful.

## Usage

```bash
# Filter categories with fewer than 80 words
uv run kozzle-word-grouper compress \
  --input-file output/word_categorization.json \
  --output-dir output/filtered \
  --min-word-count 80 \
  --no-llm-merge
```

## Results

### Before Filtering (5874 words):
- 하위개념: 11,106 categories
- 기능: 9,864 categories
- 사용맥락: 11,140 categories

### After Filtering (min_word_count=80):
- **하위개념: 3 categories** (99.97% reduction)
  - 가족구성원: 110 words
  - 시간: 105 words
  - 의류: 96 words

- **기능: 2 categories** (99.98% reduction)
  - 감정표현: 160 words
  - 시간표현: 103 words

- **사용맥락: 6 categories** (99.95% reduction)
  - 일상대화: 297 words
  - 일상생활: 234 words
  - 교육: 158 words
  - 감정표현: 156 words
  - 가족관계: 116 words
  - More...

## Use Cases

1. **Remove noise**: Filter out categories with only 1-10 words (typos, uncommon categories)
2. **Focus on common patterns**: Keep only well-established categories
3. **Simplify analysis**: Work with fewer, more meaningful categories
4. **Database optimization**: Reduce storage and improve query performance

## Recommended Values

- **min_word_count=50**: Keep more categories (~10-20 per type)
- **min_word_count=80**: Keep only largest categories (~3-6 per type)
- **min_word_count=100**: Keep only very common categories (~1-3 per type)

## Implementation

### Files Modified:
- `category_compressor.py`: Added `_filter_by_word_count()` method
- `core.py`: Added `min_word_count` parameter to `run_category_compression()`
- `cli.py`: Added `--min-word-count` / `-m` option
- `export.py`: Already compatible with filtered output
- `README.md`: Updated with documentation and examples

### Algorithm:
1. Normalize categories (remove spaces)
2. Merge exact duplicates
3. LLM semantic merging (optional)
4. Sort by word count (descending)
5. **Filter by min_word_count** (new)
6. Update statistics and export

## Performance

- **Processing time**: Negligible (~1-2 seconds added)
- **Memory impact**: Minimal (operates on existing data structures)
- **Output size**: Dramatically reduced (see results above)

## Example Commands

```bash
# Quick filter (normalize + filter, no LLM)
uv run kozzle-word-grouper compress \
  --input-file output/word_categorization.json \
  --output-dir output/filtered \
  --min-word-count 80 \
  --no-llm-merge

# Filter after LLM merging
uv run kozzle-word-grouper compress \
  --input-file output/word_categorization.json \
  --output-dir output/compressed_filtered \
  --min-word-count 80

# Aggressive filtering (keep only very large categories)
uv run kozzle-word-grouper compress \
  --input-file output/word_categorization.json \
  --output-dir output/minimal \
  --min-word-count 100 \
  --no-llm-merge
```

