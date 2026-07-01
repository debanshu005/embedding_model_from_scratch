# Embedding Model from Scratch

**Build an OpenAI-style text embedding encoder in pure PyTorch — one notebook, zero black boxes.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-Open%20Source-blue.svg)](#license)

**Repository:** [github.com/debanshu005/embedding_model_from_scratch](https://github.com/debanshu005/embedding_model_from_scratch)

---

> You call `openai.embeddings.create()` and get back a vector.  
> This project shows you **exactly** how that vector is built — transformer encoder, mean pooling, L2 norm, contrastive training on real NLI data.

Companion to **[llm_from_scratch](https://github.com/debanshu005/llm_from_scratch)** — if that repo teaches **generation** (causal GPT), this one teaches **representation** (bidirectional encoder + semantic search).

---

## Table of contents

- [Why this project](#why-this-project)
- [What you build](#what-you-build)
- [Architecture](#architecture)
- [Training objective](#training-objective)
- [Datasets](#datasets)
- [Notebook walkthrough](#notebook-walkthrough)
- [Quick start](#quick-start)
- [Default config](#default-config)
- [Results & demos](#results--demos)
- [Scale up](#scale-up)
- [LLM vs embedding encoder](#llm-vs-embedding-encoder)
- [Tech stack](#tech-stack)
- [Author](#author)

---

## Why this project

Embedding APIs power RAG, semantic search, clustering, and recommendation — but the mechanics are rarely taught hands-on.

This repo strips it down to first principles:

| You won't use | You will build |
|---------------|----------------|
| `sentence-transformers` | Bidirectional transformer encoder |
| `openai` SDK for training | Masked mean pooling + L2 normalization |
| Pre-trained embedding weights | InfoNCE + **hard-negative** triplet loss |
| Magic cosine APIs | Dot-product retrieval from scratch |

Runs on a **laptop CPU** in minutes. Every layer is readable. Every loss term is explicit.

---

## What you build

A mini version of the pipeline behind `text-embedding-ada-002` and `text-embedding-3-small`:

```
Text triplet (anchor, positive, hard negative)
        │
        ▼
   BPE tokenization (tiktoken)
        │
        ▼
   Transformer encoder (bidirectional)
        │
        ▼
   Masked mean pooling
        │
        ▼
   L2 normalize → unit vector
        │
        ▼
   Cosine-optimized training → semantic search & similarity
```

**Deliverables after running the notebook:**

- Trained encoder checkpoint (`checkpoints/embedding_encoder_scratch.pt`)
- Contrastive loss curves (train + val)
- **Before vs after** cosine similarity charts (epoch 0 → epoch 5)
- Sentence similarity benchmarks
- STS Spearman correlation vs human scores
- Semantic search demo
- t-SNE cluster visualization

---

## Architecture

```
Anchor text ──┐
Positive text ┼──► Token IDs + attention mask
Hard negative ┘
        │
        ▼
Token Embedding + Sinusoidal Positional Encoding
        │
        ▼
┌──────────────────────────────────────────┐
│  TransformerEncoderBlock  (× 4)          │
│  ┌────────────────────────────────────┐  │
│  │ LayerNorm → Bi-directional Attn    │  │
│  │           → Residual               │  │
│  │ LayerNorm → Feed-Forward (GELU)    │  │
│  │           → Residual               │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
        │
        ▼
Masked mean pooling  (ignore padding tokens)
        │
        ▼
L2 normalize  →  128-d unit vector
```

### Components

| Component | Role |
|-----------|------|
| `BidirectionalSelfAttention` | Full (non-causal) multi-head attention — every token sees every token |
| `TransformerEncoderBlock` | Pre-norm residual block: attention + MLP |
| `TextEmbeddingModel` | Full encoder: embed → blocks → pool → normalize |
| `masked_mean_pool` | OpenAI-style pooling over non-padding tokens |
| `symmetric_contrastive_loss` | InfoNCE with in-batch negatives on cosine scores |
| `triplet_margin_loss` | Hard-negative margin: push contradiction below entailment |
| `score_demo_pairs` | Fixed sentence pairs for before/after training plots |

### Mapping to OpenAI embeddings

| OpenAI (`text-embedding-3-*`) | This project |
|-------------------------------|--------------|
| Transformer encoder | `TextEmbeddingModel` |
| `cl100k_base` tokenizer | `tiktoken` GPT-2 BPE |
| Mean pooling | `masked_mean_pool` |
| L2-normalized outputs | `F.normalize(..., p=2, dim=-1)` |
| Contrastive cosine training | InfoNCE + hard-negative triplets |
| 1536 / 3072 dimensions | 128 (configurable) |

---

## Training objective

The model is **end-to-end optimized for cosine similarity**.

### 1. L2 normalization

$$\hat{\mathbf{e}} = \frac{\mathbf{e}}{\|\mathbf{e}\|_2}$$

Vectors live on the unit sphere → **dot product = cosine similarity**.

### 2. InfoNCE (in-batch negatives)

$$\mathcal{L}_{\text{InfoNCE}} = \frac{1}{2}\left[\text{CE}(S/\tau, y) + \text{CE}(S^\top/\tau, y)\right]$$

where $S_{ij} = \cos(\mathbf{a}_i, \mathbf{p}_j)$ and $y_i = i$ (matching pair on the diagonal).

### 3. Hard-negative triplet margin

From AllNLI: same premise, entailment hypothesis (positive) vs contradiction hypothesis (hard negative).

$$\mathcal{L}_{\text{triplet}} = \max\left(0,\; \cos(\mathbf{a}, \mathbf{n}) - \cos(\mathbf{a}, \mathbf{p}) + m\right)$$

### Combined loss

$$\mathcal{L} = \mathcal{L}_{\text{InfoNCE}} + \lambda \cdot \mathcal{L}_{\text{triplet}}$$

### At inference

```python
scores = query_embedding @ document_embeddings.T   # cosine rank, no extra step
```

---

## Datasets

Download once locally (~140 MB). Large files are gitignored — fetch via script.

```bash
python scripts/download_data.py
```

| Dataset | Local path | Source | Purpose |
|---------|------------|--------|---------|
| **AllNLI** | `data/all-nli-pair-class/` | [sentence-transformers/all-nli](https://huggingface.co/datasets/sentence-transformers/all-nli) | Training triplets (942k rows) |
| **STS Benchmark** | `data/stsbenchmark/` | [mteb/stsbenchmark-sts](https://huggingface.co/datasets/mteb/stsbenchmark-sts) | Eval vs human similarity scores |

### Why AllNLI?

Industry-standard for training general-purpose embeddings — used in MTEB leaderboards, `sentence-transformers`, E5, and BGE fine-tuning. We build **triplets** from rows sharing the same premise:

| Role | AllNLI label | Meaning |
|------|--------------|---------|
| Anchor | premise | Reference sentence |
| Positive | `0` (entailment) | Paraphrase / same meaning |
| Hard negative | `2` (contradiction) | Same topic, opposite meaning |

Hard negatives are harder than random in-batch negatives — the model must learn fine-grained semantic boundaries.

See [`data/README.md`](data/README.md) for more detail.

---

## Notebook walkthrough

`embedding_training.ipynb` — run top to bottom:

| Section | What happens |
|---------|--------------|
| **1. Load data** | Build hard-negative triplets from local AllNLI |
| **2. Tokenize** | BPE via `tiktoken`, padding masks for pooling |
| **3. Encoder** | Implement transformer blocks from scratch |
| **4. Cosine objective** | InfoNCE + triplet margin loss |
| **4b. Before snapshot** | Score 8 fixed demo pairs at epoch 0 |
| **5. Train** | 5 epochs, AdamW, gradient clipping |
| **5b. After plot** | Before vs after bar charts + separation gap |
| **6. Evaluate** | Sentence similarity table |
| **6b. STS benchmark** | Spearman ρ vs human judgments |
| **7. Semantic search** | Query → rank documents by dot product |
| **8. t-SNE** | Visualize sentence clusters |
| **9. Checkpoint** | Save weights + config |

---

## Quick start

### Prerequisites

- Python 3.10+
- ~2 GB disk (PyTorch + datasets)
- Jupyter

### Install & run

```bash
git clone https://github.com/debanshu005/embedding_model_from_scratch.git
cd embedding_model_from_scratch

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python scripts/download_data.py  # one-time: ~140 MB to data/

jupyter notebook embedding_training.ipynb
```

### Device support

The notebook auto-detects hardware:

```python
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
```

---

## Default config

| Parameter | Value |
|-----------|-------|
| Dataset | AllNLI triplets (entailment + contradiction) |
| Training triplets | 8,000 (sampled from 942k) |
| Vocabulary | 50,257 (GPT-2 / tiktoken) |
| Embedding dim | 128 |
| Encoder layers | 4 |
| Attention heads | 4 |
| Max sequence length | 128 tokens |
| Batch size | 64 |
| Temperature τ | 0.05 |
| Triplet margin m | 0.2 |
| Hard-negative weight λ | 1.0 |
| Optimizer | AdamW (lr=3e-4, weight decay=0.01) |
| Epochs | 5 |
| Gradient clipping | 1.0 |

Training on CPU: **~few minutes** with default settings.

---

## Results & demos

After training, the notebook produces shareable artifacts:

1. **Loss curves** — contrastive loss decreasing on train & val
2. **Before / after cosine chart** — similar pairs go up, dissimilar go down
3. **Separation gap** — mean(similar) − mean(dissimilar) widens
4. **STS Spearman** — alignment with human similarity ratings
5. **Semantic search** — `"How do language models represent text?"` → top passages
6. **t-SNE plot** — topic clusters in 2D

### Share hook (copy-ready)

> I rebuilt an OpenAI-style embedding encoder from scratch in PyTorch — transformer, mean pooling, L2 norm, InfoNCE + hard negatives on AllNLI.
>
> Companion to my LLM-from-scratch repo: one generates text, this one encodes meaning for search & RAG.
>
> github.com/debanshu005/embedding_model_from_scratch

---

## Scale up

Once the pipeline runs end-to-end, improve quality by tuning these in the notebook:

```python
MAX_TRAIN_TRIPLETS = 100_000
config = EmbeddingConfig(
    vocab_size=tokenizer.n_vocab,
    n_layer=6,
    n_head=8,
    n_embd=256,
    max_seq_len=256,
)
EPOCHS = 10
```

| Change | Effect |
|--------|--------|
| More triplets | Better generalization |
| Larger `n_embd` / `n_layer` | More representational capacity |
| Longer `max_seq_len` | Handle longer passages |
| More epochs | Lower loss (watch val for overfitting) |

---

## LLM vs embedding encoder

| | [llm_from_scratch](https://github.com/debanshu005/llm_from_scratch) | This repo |
|--|---------------------------------------------------------------------|-----------|
| **Goal** | Predict next token | Encode meaning into vectors |
| **Attention** | Causal (masked future) | Bidirectional (full context) |
| **Head** | LM head → vocab logits | Mean pool → unit vector |
| **Loss** | Cross-entropy | InfoNCE + triplet margin |
| **Output** | Generated text | Similarity scores & search ranks |
| **Use case** | Chat, completion | RAG, clustering, dedup |

Together they cover the two core primitives of modern AI apps: **retrieve** then **generate**.

---

## Project structure

```
embedding_model_from_scratch/
├── embedding_training.ipynb   # Full pipeline: data → train → evaluate
├── scripts/
│   ├── download_data.py       # Fetch AllNLI + STS locally
│   └── export_carousel_assets.py  # Generate share carousel slides
├── assets/
│   └── carousel/              # Exported slide PNGs + manifest.json
├── data/
│   └── README.md              # Dataset documentation
├── checkpoints/               # Saved weights (after training)
├── requirements.txt
└── README.md
```

---

## Tech stack

| Tool | Purpose |
|------|---------|
| [PyTorch](https://pytorch.org/) | Model, training loop, autograd |
| [tiktoken](https://github.com/openai/tiktoken) | GPT-2 BPE tokenization |
| [Hugging Face Datasets](https://huggingface.co/docs/datasets/) | Load & cache AllNLI, STS |
| [scikit-learn](https://scikit-learn.org/) | t-SNE visualization |
| [SciPy](https://scipy.org/) | Spearman correlation on STS |
| [Matplotlib](https://matplotlib.org/) | Loss curves, before/after plots |

---

## What you'll learn

- Why modern embeddings use **encoders**, not Word2Vec
- How **bidirectional** attention differs from causal GPT attention
- Why **L2 norm + dot product** is the standard retrieval trick
- How **in-batch negatives** and **hard negatives** work together
- How the same vectors power **semantic search** and **RAG** pipelines
- How to read **STS benchmarks** like the embedding papers do

---

## Author

**Debanshu Biswas**

- GitHub: [@debanshu005](https://github.com/debanshu005)
- Also built: [llm_from_scratch](https://github.com/debanshu005/llm_from_scratch)

If this helped you understand how embedding models work under the hood, consider starring the repo.

---

## License

Open source — free to use, modify, and share for learning.
