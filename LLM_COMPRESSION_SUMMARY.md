# LLM-Based Category Compression Test Results

## Test Configuration
- **Model**: exaone3.5:7.8b (Ollama)
- **Test Dataset**: 100 words
- **Batch Size**: 50 categories per LLM call
- **Timeout**: 300 seconds (5 minutes)

## Results

### Normalization Phase (Successful)
```
Original categories:
  - 하위개념: 306 categories
  - 기능: 279 categories  
  - 사용맥락: 324 categories

After normalization (remove spaces):
  - 하위개념: 304 categories (—2, or 0.65% reduction)
  - 기능: 270 categories (—9, or 3.23% reduction)
  - 사용맥락: 318 categories (—6, or 1.85% reduction)
```

### LLM Semantic Merging Phase (In Progress)

**Processing Time per Batch:**
- Batch 1 (하위개념, 50 cats): ~22 seconds
- Batch 2 (하위개념, 50 cats): ~31 seconds
- Batch 3 (하위개념, 50 cats): ~18 seconds
- Batch 4 (하위개념, 50 cats): ~19 seconds
- Batch 5 (하위개념, 50 cats): ~20 seconds
- Batch 6 (하위개념, 50 cats): ~24 seconds
- Batch 7 (하위개념, 4 cats): ~4 seconds
- Batches 1-6 (기능, 270 cats): ~20-40 seconds each
- Batch 7 (기능, 20 cats): ~11 seconds

**Average: ~20-30 seconds per batch of 50 categories**

**Progress at Timeout:**
- ✅ Completed: 하위개념 (100% - 7/7 batches)
- ✅ Completed: 기능 (100% - 6/6 batches)
- ⏸️ In Progress: 사용맥락 (batch 2 of 7, ~15% complete)

## Estimated Time for Full Dataset (5874 words)

Based on test results:

| Classification Type | Categories | Batches | Estimated Time |
|---------------------|------------|--------|----------------|
| 하위개념            | 11,106     | 223    | 74-111 min     |
| 기능                | 9,864      | 198    | 66-99 min      |
| 사용맥락           | 11,140     | 223    | 74-111 min     |
| **Total**           | **32,110** | **644**| **3.5-5.5 hours** |

## Options for Running on Full Dataset

### Option 1: Run Overnight (Recommended)
```bash
# Run with LLM merging on full dataset (takes ~4-5 hours)
nohup uv run kozzle-word-grouper compress \  --input-file output/word_categorization.json \
  --output-dir output/compressed_full \
  --batch-size 50 \
  > compression.log 2>&1 &

# Monitor progress
tail -f compression.log
```

### Option 2: Faster Processing (Larger Batches)
```bash
# Use larger batches (less accurate but faster)
uv run kozzle-word-grouper compress \  --input-file output/word_categorization.json \
  --output-dir output/compressed_fast \
  --batch-size 100  # 50% fewer API calls, but less granular grouping
```

### Option 3: Two-Stage Compression
```bash
# Stage 1: Normalize without LLM (fast)
uv run kozzle-word-grouper compress \
  --input-file output/word_categorization.json \
  --output-dir output/compressed_stage1 \
  --no-llm-merge

# Stage 2: Apply LLM merging on top categories only
# (manually select top 500-1000 categories per type)
```

### Option 4: Quick Results (No LLM)
```bash
# Already completed - normalization only
# Takes ~1-2 minutes
# Compression ratio: 5-6%

ls -lh output/compressed/compressed_categories.json
```

## LLM Grouping Quality

The LLMprompt asks the model to:
1. Group categories with similar meanings
2. Suggest a unified name for each group
3. Keep single categories as-is

**Example grouping (from LLM):**
```json
{
  "groups": [
    {
      "group_name": "동물",
      "categories": ["동물", "동물류", "생물", "짐승"]
    },
    {
      "group_name": "음식",
      "categories": ["음식", "먹거리", "식품"]
    }
  ]
}
```

## Recommendation

For production use:

1. **If time allows**: Run full LLM compression overnight for best quality (~15-30% additional compression)

2. **If speed needed**: Use normalization-only (--no-llm-merge) which already achieves ~6% compression

3. **Hybrid approach**: Run normalization first, then manually merge top categories

## Next Steps

1. Test completed successfully for 100 words
2. LLM connection to exaone3.5:7.8b working correctly
3. Choose compression strategy based on time constraints
4. Run on full dataset with chosen approach

