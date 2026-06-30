# Embeddings from Scratch

A hands-on implementation of an **OpenAI-style text embedding encoder** built entirely in PyTorch — no `sentence-transformers`, no `openai` SDK for training. This project walks through the same core ideas behind `text-embedding-ada-002` and `text-embedding-3-small`: transformer encoding, mean pooling, L2 normalization, and contrastive learning.

**Repository:** [github.com/debanshu005/embedding_model_from_scratch](https://github.com/debanshu005/embedding_model_from_scratch)

Companion project to [llm_from_scratch](https://github.com/debanshu005/llm_from_scratch): if that repo teaches *generation* (causal GPT), this one teaches *representation* (bi-directional encoder + retrieval vectors).

---

## Overview

OpenAI embedding APIs return a fixed-size vector for any text chunk. Under the hood that is **not** Word2Vec — it is a transformer encoder trained so semantically similar texts land close together. This notebook rebuilds that pipeline:

- Load entailment sentence pairs (AllNLI)
- Tokenize with BPE (`tiktoken`, same family as GPT)
- Implement a **bidirectional** transformer encoder (no causal mask)
- Pool token vectors with **masked mean pooling**
- **L2-normalize** outputs (cosine similarity = dot product)
- Train with **InfoNCE + hard-negative triplet loss** on AllNLI triplets
- Snapshot **before/after** cosine scores on fixed demo pairs
- Evaluate sentence similarity, semantic search, and t-SNE
- Save checkpoints

Default config trains on CPU in a few minutes.

---

## Architecture (OpenAI-style)

```
Text pair (anchor, positive)
        │
        ▼
BPE token IDs + attention mask
        │
        ▼
Token Embedding + Positional Encoding
        │
        ▼
┌──────────────────────────────────────┐
│  Transformer Encoder Block (× N)     │
│  ┌────────────────────────────────┐  │
│  │ LayerNorm → Bi-directional Attn│  │
│  │           → Residual           │  │
│  │ LayerNorm → Feed-Forward       │  │
│  │           → Residual           │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
        │
        ▼
Masked mean pooling over tokens
        │
        ▼
L2 normalize  →  unit vector (e.g. 128-d)
        │
        ▼
Contrastive loss: pull pairs together, push batch negatives apart
```

### How this maps to OpenAI embeddings

| OpenAI (`text-embedding-3-*`) | This project |
|-------------------------------|--------------|
| Transformer encoder | `TextEmbeddingModel` (bidirectional) |
| Subword tokenizer (cl100k_base) | `tiktoken` GPT-2 BPE |
| Mean pooling over tokens | `masked_mean_pool` |
| L2-normalized output vectors | `F.normalize(..., p=2, dim=-1)` |
| Contrastive / cosine training | InfoNCE + hard-negative triplet margin |
| 1536 / 3072 dimensions | 128 (configurable) |

---

## Data (local)

Download once — **~140 MB total**:

```bash
python scripts/download_data.py
```

| Dataset | Local path | Hugging Face | Purpose |
|---------|------------|--------------|---------|
| **AllNLI** | `data/all-nli-pair-class/` | [sentence-transformers/all-nli](https://huggingface.co/datasets/sentence-transformers/all-nli) | Training — 942k SNLI+MultiNLI pairs |
| **STS Benchmark** | `data/stsbenchmark/` | [mteb/stsbenchmark-sts](https://huggingface.co/datasets/mteb/stsbenchmark-sts) | Evaluation — human similarity scores |

### Why AllNLI?

Research and industry standard for training general-purpose embeddings (MTEB papers, sentence-transformers, E5/BGE fine-tuning stages). We use **entailment** pairs as positives for cosine contrastive learning.

See `data/README.md` for details.

---

## Cosine similarity optimization

The model is end-to-end optimized for cosine retrieval:

1. **L2 normalize** outputs → vectors live on the unit sphere
2. **Contrastive loss** uses dot products = cosine similarity on unit vectors
3. **Temperature** τ sharpens the softmax over cosine scores
4. **Inference** ranks by dot product (no extra cosine step)

```
Train:  InfoNCE(anchor, positive) + triplet_margin(anchor, positive, hard_negative)
Infer:  rank = query_embedding @ document_embeddings.T
```

## Default training config

| Parameter | Value |
|-----------|-------|
| Dataset | AllNLI entailment pairs |
| Embedding dim | 128 |
| Layers | 4 |
| Attention heads | 4 |
| Max sequence length | 128 tokens |
| Training pairs | 8,000 (sampled) |
| Batch size | 64 |
| Temperature | 0.05 |
| Optimizer | AdamW (lr=3e-4) |
| Epochs | 5 |

---

## Project structure

```
embeddings_from_scratch/
├── embedding_training.ipynb
├── scripts/download_data.py   # download AllNLI + STS locally
├── data/                      # local datasets (~140 MB, gitignored)
├── requirements.txt
├── checkpoints/
└── README.md
```

---

## Getting started

```bash
git clone https://github.com/debanshu005/embedding_model_from_scratch.git
cd embedding_model_from_scratch

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
python scripts/download_data.py   # one-time: saves data/ locally
jupyter notebook embedding_training.ipynb
```

---

## What you'll learn

- Why OpenAI embeddings use encoders, not Word2Vec
- How bidirectional attention differs from causal GPT attention
- Why mean pooling + L2 norm makes cosine search efficient
- How contrastive learning with in-batch negatives scales training
- How the same vectors power semantic search and RAG retrieval

---

## LinkedIn post angle

Artifacts from the notebook:

1. Architecture diagram (encoder → pool → normalize)
2. Contrastive loss curve
3. Sentence similarity table
4. Semantic search demo
5. t-SNE of topic clusters

Suggested hook: *"I rebuilt an OpenAI-style embedding encoder from scratch in PyTorch — transformer, mean pooling, L2 norm, contrastive loss. Companion to my LLM-from-scratch repo."*

---

## Tech stack

- [PyTorch](https://pytorch.org/)
- [tiktoken](https://github.com/openai/tiktoken) — BPE tokenization
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/) — AllNLI
- [scikit-learn](https://scikit-learn.org/) — t-SNE
- [Matplotlib](https://matplotlib.org/)

---

## Author

**Debanshu Biswas** — [GitHub](https://github.com/debanshu005)
