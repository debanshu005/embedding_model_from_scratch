#!/usr/bin/env python3
"""Export carousel slide images for sharing (demo data — no training required)."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "carousel"

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

# (x, y, label, color, annotation offset in points)
TSNE_POINTS = [
    (0.18, 0.86, "Dogs run in the park.", "#22d3ee", (14, -18)),
    (0.34, 0.74, "A puppy plays outside.", "#22d3ee", (14, 14)),
    (0.14, 0.50, "Cats sleep on the sofa.", "#a78bfa", (14, 0)),
    (0.86, 0.30, "Stock prices\ndropped sharply.", "#f472b6", (-90, 5)),
    (0.66, 0.08, "The market\ndeclined today.", "#f472b6", (14, 16)),
    (0.76, 0.80, "She studies\ncomputer science.", "#34d399", (-95, 8)),
    (0.60, 0.62, "He researches\nmachine learning.", "#34d399", (12, -20)),
    (0.40, 0.16, "Pizza is a popular\nItalian food.", "#fbbf24", (-5, -38)),
    (0.56, 0.26, "Pasta dishes come\nfrom Italy.", "#fbbf24", (12, 12)),
]

# Modern dark theme with vibrant accents.
C = {
    "bg": "#0f172a",
    "panel": "#1e293b",
    "grid": "#334155",
    "text": "#f1f5f9",
    "muted": "#94a3b8",
    "before": "#c084fc",
    "after": "#22d3ee",
    "train": "#34d399",
    "val": "#f472b6",
    "steps": ["#22d3ee", "#a78bfa", "#f472b6", "#34d399", "#fbbf24"],
    "rank": ["#22d3ee", "#a78bfa", "#fbbf24"],
}

RC = {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"]}


def _new_fig(w: float, h: float, *, top: float = 0.90):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=C["bg"])
    fig.subplots_adjust(top=top, left=0.10, right=0.96, bottom=0.12)
    ax.set_facecolor(C["bg"])
    return fig, ax


def _header(fig, title: str, subtitle: str | None = None) -> None:
    fig.text(0.06, 0.97, title, fontsize=15, fontweight="bold", color=C["text"], ha="left", va="top")
    if subtitle:
        fig.text(0.06, 0.925, subtitle, fontsize=10, color=C["muted"], ha="left", va="top")


def _style_ax(ax) -> None:
    ax.tick_params(colors=C["muted"], labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(C["grid"])
    ax.xaxis.label.set_color(C["muted"])
    ax.yaxis.label.set_color(C["muted"])


def _legend(ax) -> None:
    leg = ax.legend(frameon=True, facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"])
    for text in leg.get_texts():
        text.set_color(C["text"])


def _save(fig: plt.Figure, name: str) -> Path:
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")
    return path


def _rounded_box(ax, xy, w, h, fc, ec, text, text_color=C["text"], fontsize=10):
    box = FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.08,rounding_size=0.15",
        facecolor=fc, edgecolor=ec, linewidth=2, alpha=0.95,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color)


def architecture_diagram() -> None:
    fig, ax = _new_fig(11.5, 6.5, top=1.0)
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    _header(fig, "OpenAI-style embedding pipeline", "From raw text to searchable vectors — built in PyTorch")

    steps = [
        "Text triplet\n(anchor, +, −)",
        "BPE tokenize\n(tiktoken)",
        "Transformer\nencoder",
        "Mean pool\n+ L2 norm",
        "Cosine search\n& RAG",
    ]
    box_w, box_h, gap = 1.45, 1.85, 0.42
    total_w = len(steps) * box_w + (len(steps) - 1) * gap
    x0 = (11.5 - total_w) / 2
    y_box = 2.15

    for i, (label, color) in enumerate(zip(steps, C["steps"])):
        x = x0 + i * (box_w + gap)
        _rounded_box(ax, (x, y_box), box_w, box_h, fc=color + "33", ec=color, text=label, fontsize=9)
        if i < len(steps) - 1:
            x_next = x0 + (i + 1) * (box_w + gap)
            arrow = FancyArrowPatch(
                (x + box_w + 0.04, y_box + box_h / 2),
                (x_next - 0.04, y_box + box_h / 2),
                arrowstyle="-|>", mutation_scale=14, lw=2.5, color=C["steps"][i + 1], alpha=0.9,
            )
            ax.add_patch(arrow)

    pill_w, pill_h = 7.2, 0.9
    ax.add_patch(FancyBboxPatch(
        ((11.5 - pill_w) / 2, 0.45), pill_w, pill_h,
        boxstyle="round,pad=0.1,rounding_size=0.2",
        facecolor=C["panel"], edgecolor=C["after"], linewidth=1.5, alpha=0.9,
    ))
    ax.text(5.75, 0.90, "InfoNCE + hard-negative triplet loss  ·  AllNLI", ha="center",
            fontsize=11, color=C["after"], fontweight="bold")
    _save(fig, "01_architecture.png")


def before_after_cosine() -> None:
    fig, ax = _new_fig(12, 5.8, top=0.86)
    _header(fig, "Before vs after training", "Similar pairs rise · dissimilar pairs fall")
    x = np.arange(len(BEFORE_VALS))
    w = 0.36
    ax.bar(x - w / 2, BEFORE_VALS, w, label="before (epoch 0)", color=C["before"], edgecolor="white", linewidth=0.6, alpha=0.9)
    ax.bar(x + w / 2, AFTER_VALS, w, label="after (epoch 5)", color=C["after"], edgecolor="white", linewidth=0.6, alpha=0.95)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i + 1}\n({k})" for i, k in enumerate(PAIR_KINDS)], fontsize=9)
    ax.set_ylabel("Cosine similarity")
    ax.set_xlabel("Demo pair")
    ax.set_ylim(0, 1.08)
    ax.grid(True, axis="y", color=C["grid"], alpha=0.5, linestyle="--")
    _style_ax(ax)
    _legend(ax)
    _save(fig, "02_before_after_cosine.png")


def separation_gap() -> None:
    labels = ["similar", "dissimilar", "gap\n(sim − dis)"]
    before = [BEFORE_SUMMARY["similar"], BEFORE_SUMMARY["dissimilar"], BEFORE_SUMMARY["gap"]]
    after = [AFTER_SUMMARY["similar"], AFTER_SUMMARY["dissimilar"], AFTER_SUMMARY["gap"]]
    fig, ax = _new_fig(8, 5.2, top=0.86)
    _header(fig, "Separation gap widens", f"After training: gap = {AFTER_SUMMARY['gap']:.2f}")
    x = np.arange(len(labels))
    w = 0.34
    ax.bar(x - w / 2, before, w, label="before", color=C["before"], edgecolor="white", linewidth=0.6, alpha=0.9)
    bars_a = ax.bar(x + w / 2, after, w, label="after", color=C["after"], edgecolor="white", linewidth=0.6, alpha=0.95)
    for bar, color in zip(bars_a, [C["train"], C["val"], C["steps"][4]]):
        bar.set_facecolor(color)
    ax.axhline(0, color=C["grid"], linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean cosine")
    ax.grid(True, axis="y", color=C["grid"], alpha=0.5, linestyle="--")
    _style_ax(ax)
    _legend(ax)
    _save(fig, "03_separation_gap.png")


def tsne_clusters() -> None:
    fig, ax = _new_fig(11, 8, top=0.88)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.86, bottom=0.06)
    _header(fig, "Topic clusters in embedding space", "t-SNE projection (demo)")
    ax.set_facecolor(C["panel"])

    for x, y, label, color, (ox, oy) in TSNE_POINTS:
        ax.scatter([x], [y], s=320, alpha=0.22, color=color, linewidths=0, zorder=2)
        ax.scatter([x], [y], s=100, alpha=0.95, color=color, edgecolors="white", linewidths=1.2, zorder=3)
        ax.annotate(
            label, (x, y), fontsize=8.5, color=C["text"], ha="left", va="center",
            xytext=(ox, oy), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=C["bg"], edgecolor=color, alpha=0.92, linewidth=1),
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8, shrinkA=5, shrinkB=5),
            zorder=4,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, color=C["grid"], alpha=0.35, linestyle=":")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(C["grid"])
    _style_ax(ax)
    _save(fig, "04_tsne_clusters.png")


def semantic_search_card() -> None:
    fig, ax = _new_fig(10, 7.2, top=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _header(fig, "Semantic search", "Cosine similarity on L2-normalized vectors")

    card = FancyBboxPatch(
        (0.05, 0.06), 0.90, 0.78,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=C["panel"], edgecolor=C["grid"], linewidth=1.5,
    )
    ax.add_patch(card)

    ax.text(0.09, 0.78, "Query", fontsize=9, color=C["muted"], fontweight="bold")
    ax.text(
        0.09, 0.72, textwrap.fill(f'"{SEARCH_QUERY}"', width=58),
        fontsize=11, color=C["after"], fontweight="bold", va="top", linespacing=1.35,
    )

    row_tops = [0.60, 0.40, 0.22]
    for i, ((score, doc), y_top, rank_color) in enumerate(zip(SEARCH_HITS, row_tops, C["rank"])):
        ax.add_patch(FancyBboxPatch(
            (0.09, y_top - 0.055), 0.045, 0.065,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor=rank_color, edgecolor="none", alpha=0.9,
        ))
        ax.text(0.112, y_top - 0.022, str(i + 1), ha="center", va="center", fontsize=10,
                fontweight="bold", color=C["bg"])
        ax.text(0.22, y_top - 0.015, f"{score:.3f}", fontsize=11, color=rank_color,
                fontweight="bold", family="monospace", va="center")
        ax.text(
            0.32, y_top - 0.015, textwrap.fill(doc, width=46),
            fontsize=9.5, color=C["text"], va="top", linespacing=1.3,
        )

    _save(fig, "05_semantic_search.png")


def loss_curve() -> None:
    fig, ax = _new_fig(7, 5.2, top=0.86)
    _header(fig, "Contrastive training converges", "InfoNCE + triplet margin on AllNLI")
    epochs = np.arange(1, len(TRAIN_LOSS) + 1)
    ax.fill_between(epochs, TRAIN_LOSS, alpha=0.2, color=C["train"])
    ax.fill_between(epochs, VAL_LOSS, alpha=0.15, color=C["val"])
    ax.plot(epochs, TRAIN_LOSS, marker="o", markersize=8, label="train", color=C["train"],
            linewidth=2.5, markeredgecolor="white", markeredgewidth=1)
    ax.plot(epochs, VAL_LOSS, marker="o", markersize=8, label="val", color=C["val"],
            linewidth=2.5, markeredgecolor="white", markeredgewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Contrastive loss")
    ax.set_xlim(0.8, 5.2)
    ax.grid(True, color=C["grid"], alpha=0.5, linestyle="--")
    _style_ax(ax)
    _legend(ax)
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
        "theme": "dark-modern",
        "note": "Demo assets — representative values from notebook run",
    }
    path = OUT / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  saved {path.relative_to(ROOT)}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(RC)
    print(f"Exporting carousel slides to {OUT.relative_to(ROOT)}/")
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
