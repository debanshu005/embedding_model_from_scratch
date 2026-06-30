# Local datasets

Training and evaluation data for `embedding_training.ipynb`.

Run from project root:

```bash
python scripts/download_data.py
```

## Files

| Folder | Hugging Face source | Rows | Purpose |
|--------|---------------------|------|---------|
| `all-nli-pair-class/` | [sentence-transformers/all-nli](https://huggingface.co/datasets/sentence-transformers/all-nli) | 942k train | Triplet training with hard negatives |
| `stsbenchmark/` | [mteb/stsbenchmark-sts](https://huggingface.co/datasets/mteb/stsbenchmark-sts) | 1.5k test | Similarity evaluation (human scores 0–5) |

### Why AllNLI?

AllNLI is the most widely used open dataset for training general-purpose sentence embeddings. The notebook builds **triplets** from rows sharing the same premise:

- **Positive** — entailment hypothesis (`label=0`)
- **Hard negative** — contradiction hypothesis (`label=2`)

This is stronger than in-batch negatives alone because the negative is about the *same topic* but with opposite meaning.

### Why STS Benchmark?

STS provides human-annotated similarity scores. After training, we measure whether our model's cosine scores correlate with human judgment (Spearman correlation).

**Note:** `data/` is gitignored (~150 MB). Re-download with the script above.
