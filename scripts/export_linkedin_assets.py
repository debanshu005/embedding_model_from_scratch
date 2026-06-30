#!/usr/bin/env python3
"""Export simple LinkedIn carousel images (demo data — no training required)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "linkedin"

# Representative numbers from a full notebook run (demonstration only).
BEFORE_SUMMARY = {"similar": 0.986, "dissimilar": 0.987, "gap": -0.001}
AFTER_SUMMARY = {"similar": 0.861, "dissimilar": 0.539, "gap": 0.322}
BEFORE_VALS = [0.99, 0.98, 0.99, 0.99, 0.98, 0.99, 0.98, 0.99]
AFTER_VALS = [0.92, 0.41, 0.88, 0.75, 0.35, 0.90, 0.38, 0.87]
PAIR_KINDS = ["sim", "dis", "sim", "sim", "dis", "sim", "dis", "sim"]
TRAIN_LOSS = [4.65, 3.82, 3.21, 2.78, 2.45]
VAL_LOSS = [4.58, 3.75, 3.18, 2.81, 2.52]

SEARCH_QUERY = "How do language models represent text?"
SEARCH_HITS = [
    (0.847, "Embeddings map text into dense vector space for search."),
    (0.812, "Transformers revolutionized natural language processing."),
    (0.691, "The basketball team won the championship game."),
]

TSNE_CLUSTERS = {
    "pets": (["Dogs run in the park.", "A puppy plays outside."], (0.2, 0.75)),
    "cats": (["Cats sleep on the sofa."], (0.15, 0.55)),
    "finance": (["Stock prices dropped sharply.", "The market declined today."], (0.8, 0.25)),
    "tech": (["She studies computer science.", "He researches machine learning."], (0.75, 0.55)),
    "food": (["Pizza is a popular Italian food.", "Pasta dishes come from Italy."], (0.45, 0.2)),
}

STYLE = {"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 11}


def _save(fig: plt.Figure, name: str) -> Path:
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")
    return path


def architecture_diagram() -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("OpenAI-style embedding pipeline", fontsize=16, fontweight="bold", pad=12)

    steps = [
        "Text triplet\n(anchor, +, −)",
        "BPE tokenize\n(tiktoken)",
        "Transformer\nencoder",
        "Mean pool\n+ L2 norm",
        "Cosine search\n& RAG",
    ]
    xs = np.linspace(0.8, 9.2, len(steps))
    for i, (x, label) in enumerate(zip(xs, steps)):
        ax.add_patch(plt.Rectangle((x - 0.75, 2.0), 1.5, 1.6, fc="#e0e7ff", ec="#2563eb", lw=2))
        ax.text(x, 2.8, label, ha="center", va="center", fontsize=10, fontweight="bold")
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.85, 2.8), xytext=(x + 0.85, 2.8),
                        arrowprops=dict(arrowstyle="->", color="#64748b", lw=2))

    ax.text(5, 0.9, "InfoNCE + hard-negative triplet loss on AllNLI", ha="center", fontsize=11, color="#475569")
    _save(fig, "01_architecture.png")


def before_after_cosine() -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(BEFORE_VALS))
    w = 0.35
    ax.bar(x - w / 2, BEFORE_VALS, w, label="before (epoch 0)", color="#94a3b8")
    ax.bar(x + w / 2, AFTER_VALS, w, label="after (epoch 5)", color="#2563eb")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i + 1}\n({k})" for i, k in enumerate(PAIR_KINDS)], fontsize=9)
    ax.set_ylabel("Cosine similarity")
    ax.set_xlabel("Demo pair")
    ax.set_title("Before vs after training on fixed sentence pairs", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "02_before_after_cosine.png")


def separation_gap() -> None:
    labels = ["similar", "dissimilar", "gap (sim − dis)"]
    before = [BEFORE_SUMMARY["similar"], BEFORE_SUMMARY["dissimilar"], BEFORE_SUMMARY["gap"]]
    after = [AFTER_SUMMARY["similar"], AFTER_SUMMARY["dissimilar"], AFTER_SUMMARY["gap"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, before, w, label="before", color="#94a3b8")
    ax.bar(x + w / 2, after, w, label="after", color="#2563eb")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean cosine")
    ax.set_title("Similar vs dissimilar separation widens after training", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "03_separation_gap.png")


def tsne_clusters() -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#2563eb", "#7c3aed", "#dc2626", "#059669", "#d97706"]
    for (color, (sentences, (cx, cy))) in zip(colors, TSNE_CLUSTERS.values()):
        rng = np.random.default_rng(42)
        n = len(sentences)
        xs = cx + rng.normal(0, 0.04, n)
        ys = cy + rng.normal(0, 0.04, n)
        ax.scatter(xs, ys, s=120, alpha=0.85, color=color)
        for xi, yi, sent in zip(xs, ys, sentences):
            label = sent[:28] + ("..." if len(sent) > 28 else "")
            ax.annotate(label, (xi, yi), fontsize=8, xytext=(4, 4), textcoords="offset points")

    ax.set_title("Sentence embeddings cluster by topic (t-SNE demo)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.set_xticks([])
    ax.set_yticks([])
    _save(fig, "04_tsne_clusters.png")


def semantic_search_card() -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis("off")
    ax.set_title("Semantic search (cosine on unit vectors)", fontsize=13, fontweight="bold", loc="left")

    lines = [f'Query: "{SEARCH_QUERY}"', ""]
    for rank, (score, doc) in enumerate(SEARCH_HITS, 1):
        lines.append(f"{rank}. {score:.3f} — {doc}")

    ax.text(
        0.02, 0.92, "\n".join(lines), transform=ax.transAxes,
        fontsize=11, family="monospace", va="top",
        bbox=dict(boxstyle="round", facecolor="#f8fafc", edgecolor="#cbd5e1", pad=0.8),
    )
    _save(fig, "05_semantic_search.png")


def loss_curve() -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = np.arange(1, len(TRAIN_LOSS) + 1)
    ax.plot(epochs, TRAIN_LOSS, marker="o", label="train", color="#2563eb")
    ax.plot(epochs, VAL_LOSS, marker="o", label="val", color="#94a3b8")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Contrastive loss")
    ax.set_title("OpenAI-style contrastive training", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "06_loss_curve.png")


def write_manifest() -> None:
    manifest = {
        "carousel_order": [
            "01_architecture.png",
            "02_before_after_cosine.png",
            "03_separation_gap.png",
            "04_tsne_clusters.png",
            "05_semantic_search.png",
            "06_loss_curve.png",
        ],
        "separation_gap_after": AFTER_SUMMARY["gap"],
        "query": SEARCH_QUERY,
        "note": "Demo assets — representative values from notebook run",
    }
    path = OUT / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  saved {path.relative_to(ROOT)}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    print(f"Exporting LinkedIn carousel to {OUT.relative_to(ROOT)}/")
    architecture_diagram()
    before_after_cosine()
    separation_gap()
    tsne_clusters()
    semantic_search_card()
    loss_curve()
    write_manifest()
    print("Done — upload images in manifest carousel_order.")


if __name__ == "__main__":
    main()
