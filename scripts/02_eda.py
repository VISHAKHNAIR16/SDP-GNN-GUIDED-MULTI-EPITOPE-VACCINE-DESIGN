"""
02_eda.py
=========
Phase 2: Exploratory Data Analysis
GNN-Guided Multi-Epitope Vaccine Design

What this script produces (saved to outputs/figures/):
    01_epitope_length_distribution.png   — length histogram pos vs neg
    02_top_source_proteins.png           — which TB proteins epitopes come from
    03_iedb_vjdb_overlap.png             — how many epitopes have TCR evidence
    04_hla_allele_coverage.png           — HLA gene family distribution
    05_amino_acid_frequency.png          — AA composition pos vs neg
    06_class_imbalance.png               — pos vs neg counts + imbalance ratio
    07_epitope_position_heatmap.png      — AA preference at each position

Run from project root:
    uv run python scripts/02_eda.py
"""

from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from Bio import SeqIO
from loguru import logger
from rich.console import Console
from rich.table import Table

# ── Setup ─────────────────────────────────────────────────────────────────────

console = Console()

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR   = PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Publication-ready style ───────────────────────────────────────────────────
# These settings make figures suitable for a research paper (300 DPI, clean fonts)

plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "legend.frameon":    False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "#FAFAFA",
})

# Consistent color palette throughout
COL_POS  = "#2E86AB"   # blue  — positive/immunogenic
COL_NEG  = "#E84855"   # red   — negative/non-immunogenic
COL_HLA  = "#3BB273"   # green — HLA
COL_PROT = "#7B4F9E"   # purple — TB proteins
COL_TCR  = "#F4A261"   # orange — TCR/VDJdb


def save(fig: plt.Figure, name: str) -> None:
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved: {path.relative_to(PROJECT_ROOT)}")


# ── Load cleaned data ─────────────────────────────────────────────────────────

def load_data() -> dict:
    logger.info("Loading cleaned datasets from data/processed/")

    df_pos  = pd.read_csv(PROCESSED_DIR / "iedb_positive_clean.csv")
    df_neg  = pd.read_csv(PROCESSED_DIR / "iedb_negative_clean.csv")
    df_vjdb = pd.read_csv(PROCESSED_DIR / "vjdb_tb_human_clean.tsv", sep="\t")
    df_tb   = pd.read_csv(PROCESSED_DIR / "tb_proteome_metadata.csv")

    # Combined dataframe for analyses that need both pos + neg
    df_all = pd.concat([df_pos, df_neg], ignore_index=True)

    logger.info(f"  Positive epitopes : {len(df_pos):,}")
    logger.info(f"  Negative epitopes : {len(df_neg):,}")
    logger.info(f"  VDJdb TCR pairs   : {len(df_vjdb):,}")
    logger.info(f"  TB proteins       : {len(df_tb):,}")

    return {
        "pos":  df_pos,
        "neg":  df_neg,
        "all":  df_all,
        "vjdb": df_vjdb,
        "tb":   df_tb,
    }


# ── Plot 1: Epitope Length Distribution ───────────────────────────────────────

def plot_length_distribution(data: dict) -> None:
    """
    Why: T-cell epitopes come in two biological flavours:
      - MHC Class I  (CD8+ T-cells): 8–11 aa — you expect a spike at 9
      - MHC Class II (CD4+ T-cells): 13–25 aa — broader hump at 13–17

    A good vaccine should cover BOTH classes for maximum immune activation.
    This plot tells us the composition of our training data.
    """
    logger.info("Plot 1: Epitope length distribution")

    df_pos = data["pos"]
    df_neg = data["neg"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Epitope Length Distribution", fontsize=14, fontweight="bold", y=1.02)

    # ── Left: Overlapping histogram ──
    ax = axes[0]
    bins = range(8, 27)
    ax.hist(df_pos["seq_length"], bins=bins, alpha=0.7, color=COL_POS,
            label=f"Positive (n={len(df_pos):,})", edgecolor="white", linewidth=0.5)
    ax.hist(df_neg["seq_length"], bins=bins, alpha=0.5, color=COL_NEG,
            label=f"Negative (n={len(df_neg):,})", edgecolor="white", linewidth=0.5)

    ax.axvspan(8, 11.5,  alpha=0.08, color=COL_POS, label="MHC I range (8–11 aa)")
    ax.axvspan(11.5, 25, alpha=0.05, color=COL_NEG, label="MHC II range (12–25 aa)")

    ax.set_xlabel("Epitope length (amino acids)")
    ax.set_ylabel("Count")
    ax.set_title("Raw counts")
    ax.legend(fontsize=9)
    ax.set_xticks(range(8, 27, 2))

    # ── Right: Percentage stacked for each length ──
    ax2 = axes[1]
    lengths = sorted(set(df_pos["seq_length"]) | set(df_neg["seq_length"]))
    pos_counts = df_pos["seq_length"].value_counts()
    neg_counts = df_neg["seq_length"].value_counts()

    pos_pct, neg_pct = [], []
    for l in lengths:
        p = pos_counts.get(l, 0)
        n = neg_counts.get(l, 0)
        total = p + n
        pos_pct.append(100 * p / total if total > 0 else 0)
        neg_pct.append(100 * n / total if total > 0 else 0)

    x = np.arange(len(lengths))
    ax2.bar(x, pos_pct, color=COL_POS, alpha=0.8, label="% Positive", edgecolor="white", linewidth=0.5)
    ax2.bar(x, neg_pct, bottom=pos_pct, color=COL_NEG, alpha=0.8,
            label="% Negative", edgecolor="white", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(lengths, rotation=45)
    ax2.set_xlabel("Epitope length (amino acids)")
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("Positive vs Negative ratio by length")
    ax2.legend(fontsize=9)
    ax2.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    save(fig, "01_epitope_length_distribution.png")


# ── Plot 2: Top Source Proteins ───────────────────────────────────────────────

def plot_source_proteins(data: dict) -> None:
    """
    Why: Not all TB proteins are equally immunogenic.
    A small number of "immunodominant" proteins generate most of the
    immune response. Identifying these is a key finding for vaccine design.

    The source molecule column tells us which TB protein each epitope
    came from. We look at this for positives only.
    """
    logger.info("Plot 2: Top source proteins")

    df_pos = data["pos"]

    # Find the protein source column
    prot_col = next(
        (c for c in df_pos.columns if "source_molecule" in c or "molecule" in c),
        None
    )

    if prot_col is None or df_pos[prot_col].isna().all():
        logger.warning("  No source molecule column found — skipping plot 2")
        return

    # Clean protein names: strip long suffixes
    proteins = (
        df_pos[prot_col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.replace(r"\s*\(.*\)", "", regex=True)   # remove parenthetical
        .str.replace(r"\s*precursor.*", "", regex=True, flags=2)
        .str[:55]                                     # truncate long names
    )

    top_proteins = proteins.value_counts().head(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [COL_PROT if i < 5 else "#B8A0CC" for i in range(len(top_proteins))]
    bars = ax.barh(
        y=range(len(top_proteins)),
        width=top_proteins.values,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_yticks(range(len(top_proteins)))
    ax.set_yticklabels(top_proteins.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of immunogenic epitopes")
    ax.set_title(
        "Top 20 M. tuberculosis source proteins\n(by number of immunogenic epitopes)",
        fontweight="bold"
    )

    # Add value labels on bars
    for bar, val in zip(bars, top_proteins.values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=8, color="#333")

    # Annotation: cumulative coverage
    top5_total  = top_proteins.values[:5].sum()
    total       = len(df_pos[prot_col].dropna())
    pct_top5    = 100 * top5_total / total if total > 0 else 0
    ax.annotate(
        f"Top 5 proteins account for\n{pct_top5:.1f}% of all epitopes",
        xy=(top_proteins.values[4], 4),
        xytext=(top_proteins.values[0] * 0.6, 8),
        arrowprops=dict(arrowstyle="->", color="#333", lw=0.8),
        fontsize=9, color="#333",
    )

    plt.tight_layout()
    save(fig, "02_top_source_proteins.png")


# ── Plot 3: IEDB–VDJdb Overlap ────────────────────────────────────────────────

def plot_overlap(data: dict) -> None:
    """
    Why: Epitopes that appear in BOTH IEDB (lab-confirmed immunogenic)
    AND VDJdb (TCR recognition confirmed) are our GOLD STANDARD nodes
    in the graph — they have two independent lines of experimental evidence.

    This plot quantifies how much gold standard data we have.
    """
    logger.info("Plot 3: IEDB–VDJdb overlap")

    df_pos  = data["pos"]
    df_vjdb = data["vjdb"]

    pos_seqs  = set(df_pos["epitope_seq"].str.upper())
    vjdb_seqs = set(df_vjdb["epitope"].astype(str).str.upper())

    overlap       = pos_seqs & vjdb_seqs
    pos_only      = pos_seqs - vjdb_seqs
    vjdb_only     = vjdb_seqs - pos_seqs

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("IEDB Immunogenic Epitopes vs VDJdb TCR Evidence",
                 fontsize=14, fontweight="bold")

    # ── Left: Venn-style bar chart ──
    ax = axes[0]
    categories  = ["IEDB only\n(no TCR data)", "Both IEDB\n+ VDJdb", "VDJdb only\n(not in IEDB)"]
    counts      = [len(pos_only), len(overlap), len(vjdb_only)]
    colors      = [COL_POS, "#2D9B6F", COL_TCR]
    bars        = ax.bar(categories, counts, color=colors, edgecolor="white",
                         linewidth=0.5, width=0.5)

    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Number of unique epitopes")
    ax.set_title("Epitope overlap between databases")

    # ── Right: What the overlap means ──
    ax2 = axes[1]
    ax2.axis("off")

    explanation = (
        f"Database statistics\n"
        f"{'─' * 38}\n\n"
        f"IEDB immunogenic epitopes : {len(pos_seqs):,}\n"
        f"VDJdb TB epitopes          : {len(vjdb_seqs):,}\n"
        f"Overlap (gold standard)    : {len(overlap):,}\n\n"
        f"{'─' * 38}\n\n"
        f"Gold standard epitopes are those\n"
        f"confirmed by TWO independent\n"
        f"experimental methods:\n\n"
        f"  1. T-cell activation assay (IEDB)\n"
        f"  2. TCR binding confirmed (VDJdb)\n\n"
        f"These {len(overlap)} epitopes will have the\n"
        f"strongest edges in the GNN graph."
    )

    ax2.text(0.05, 0.95, explanation, transform=ax2.transAxes,
             fontsize=10, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0F7FF",
                       edgecolor=COL_POS, linewidth=1))

    if overlap:
        logger.info(f"  Gold standard epitopes: {sorted(overlap)[:5]} ...")

    plt.tight_layout()
    save(fig, "03_iedb_vjdb_overlap.png")

    return overlap


# ── Plot 4: HLA Allele Coverage ───────────────────────────────────────────────

def plot_hla_coverage(data: dict) -> None:
    """
    Why: A vaccine should work across diverse human populations.
    Different ethnic groups have different dominant HLA alleles.
    This plot shows the distribution of HLA gene families in our data,
    telling us how broad our potential population coverage is.
    """
    logger.info("Plot 4: HLA allele coverage")

    hla_path = PROJECT_ROOT / "data" / "processed" / "hla_prot_clean.fasta"
    if not hla_path.exists():
        logger.warning("  HLA FASTA not found — skipping")
        return

    import re
    gene_counts = Counter()
    for rec in SeqIO.parse(str(hla_path), "fasta"):
        m = re.search(r"\b([A-Z0-9]+)\*\d+:\d+", rec.description.upper())
        gene = m.group(1) if m else "Other"
        gene_counts[gene] += 1

    # Group into families for cleaner plot
    family_map = {
        "A": "HLA-A (Class I)",
        "B": "HLA-B (Class I)",
        "C": "HLA-C (Class I)",
        "DRA": "HLA-DR (Class II)", "DRB1": "HLA-DR (Class II)",
        "DRB3": "HLA-DR (Class II)", "DRB4": "HLA-DR (Class II)",
        "DRB5": "HLA-DR (Class II)",
        "DQA1": "HLA-DQ (Class II)", "DQB1": "HLA-DQ (Class II)",
        "DPA1": "HLA-DP (Class II)", "DPB1": "HLA-DP (Class II)",
    }
    family_counts = Counter()
    for gene, cnt in gene_counts.items():
        family = family_map.get(gene, "Other/Non-classical")
        family_counts[family] += cnt

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("HLA Allele Coverage in Cleaned Dataset",
                 fontsize=14, fontweight="bold")

    # ── Left: Pie chart ──
    ax = axes[0]
    labels = list(family_counts.keys())
    sizes  = list(family_counts.values())
    hla_colors = [
        "#2E86AB", "#1A5E80", "#73B9D4",   # Blues for Class I
        "#3BB273", "#2A8050", "#5CC98A", "#A0DDC0",  # Greens for Class II
        "#AAAAAA"
    ][:len(labels)]

    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, colors=hla_colors,
        autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(linewidth=0.5, edgecolor="white")
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.legend(wedges, [f"{l} ({v:,})" for l, v in zip(labels, sizes)],
              loc="lower left", fontsize=8)
    ax.set_title("HLA gene family distribution")

    # ── Right: Class I vs Class II breakdown ──
    ax2 = axes[1]
    class1 = sum(v for k, v in family_counts.items() if "Class I" in k)
    class2 = sum(v for k, v in family_counts.items() if "Class II" in k)
    other  = sum(v for k, v in family_counts.items() if "Class" not in k)

    ax2.bar(["MHC Class I\n(CD8+ T-cells)", "MHC Class II\n(CD4+ T-cells)", "Other"],
            [class1, class2, other],
            color=[COL_POS, COL_HLA, "#AAAAAA"],
            edgecolor="white", linewidth=0.5, width=0.5)

    for i, v in enumerate([class1, class2, other]):
        ax2.text(i, v + 50, f"{v:,}", ha="center", fontweight="bold")

    ax2.set_ylabel("Number of alleles")
    ax2.set_title("Class I vs Class II breakdown\n(determines which T-cells respond)")

    ax2.annotate(
        "Class I → CD8+ killer T-cells\nClass II → CD4+ helper T-cells\nBoth needed for a strong vaccine",
        xy=(1, class2), xytext=(1.3, class2 * 0.6),
        fontsize=9, color="#444",
        arrowprops=dict(arrowstyle="->", color="#888", lw=0.6),
    )

    plt.tight_layout()
    save(fig, "04_hla_allele_coverage.png")


# ── Plot 5: Amino Acid Frequency ──────────────────────────────────────────────

def plot_amino_acid_frequency(data: dict) -> None:
    """
    Why: Immunogenic peptides have distinct amino acid compositions.
    HLA molecules have specific 'anchor positions' (usually position 2
    and the C-terminal position) where they grip the peptide.
    Certain amino acids (hydrophobic: L, I, V, M) preferentially appear
    at these positions in immunogenic peptides.

    This plot reveals biochemical differences between pos and neg epitopes.
    """
    logger.info("Plot 5: Amino acid frequency")

    AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")

    def aa_freq(sequences: pd.Series) -> dict:
        counter = Counter()
        total = 0
        for seq in sequences:
            for aa in str(seq).upper():
                if aa in AA_ORDER:
                    counter[aa] += 1
                    total += 1
        return {aa: counter[aa] / total * 100 for aa in AA_ORDER} if total > 0 else {}

    pos_freq = aa_freq(data["pos"]["epitope_seq"])
    neg_freq = aa_freq(data["neg"]["epitope_seq"])

    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    fig.suptitle("Amino Acid Composition: Immunogenic vs Non-immunogenic TB Epitopes",
                 fontsize=14, fontweight="bold")

    x = np.arange(len(AA_ORDER))
    width = 0.35

    # ── Top: Side-by-side bar chart ──
    ax = axes[0]
    pos_vals = [pos_freq.get(aa, 0) for aa in AA_ORDER]
    neg_vals = [neg_freq.get(aa, 0) for aa in AA_ORDER]

    ax.bar(x - width/2, pos_vals, width, color=COL_POS, alpha=0.85,
           label="Positive (immunogenic)", edgecolor="white", linewidth=0.3)
    ax.bar(x + width/2, neg_vals, width, color=COL_NEG, alpha=0.85,
           label="Negative", edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(AA_ORDER)
    ax.set_ylabel("Frequency (%)")
    ax.set_title("Amino acid frequency in positive vs negative epitopes")
    ax.legend()

    # ── Bottom: Enrichment ratio (pos/neg) ──
    ax2 = axes[1]
    enrichment = []
    for aa in AA_ORDER:
        p = pos_freq.get(aa, 0.001)
        n = neg_freq.get(aa, 0.001)
        enrichment.append(np.log2(p / n))

    colors_enrich = [COL_POS if e > 0 else COL_NEG for e in enrichment]
    ax2.bar(x, enrichment, color=colors_enrich, alpha=0.85, edgecolor="white", linewidth=0.3)
    ax2.axhline(0, color="gray", linewidth=0.8)
    ax2.axhline(0.5,  color=COL_POS, linewidth=0.6, linestyle="--", alpha=0.5)
    ax2.axhline(-0.5, color=COL_NEG, linewidth=0.6, linestyle="--", alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(AA_ORDER)
    ax2.set_ylabel("Log₂ enrichment (pos/neg)")
    ax2.set_title("Enrichment in immunogenic epitopes (positive = enriched, negative = depleted)")

    # Label the most enriched/depleted
    for i, (aa, val) in enumerate(zip(AA_ORDER, enrichment)):
        if abs(val) > 0.4:
            ax2.text(i, val + (0.05 if val > 0 else -0.1),
                     aa, ha="center", fontsize=9, fontweight="bold",
                     color="#333")

    plt.tight_layout()
    save(fig, "05_amino_acid_frequency.png")


# ── Plot 6: Class Imbalance ───────────────────────────────────────────────────

def plot_class_imbalance(data: dict) -> None:
    """
    Why: The model must know the ratio of positives to negatives.
    A 9:1 imbalance means naive models predict "negative" always and look
    90% accurate. We must use weighted loss or oversampling to fix this.
    This plot documents the imbalance for the paper's methods section.
    """
    logger.info("Plot 6: Class imbalance")

    n_pos = len(data["pos"])
    n_neg = len(data["neg"])
    ratio = n_neg / n_pos

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Class Imbalance in Training Data", fontsize=14, fontweight="bold")

    # ── Left: Bar chart ──
    ax = axes[0]
    bars = ax.bar(
        ["Positive\n(immunogenic)", "Negative\n(non-immunogenic)"],
        [n_pos, n_neg],
        color=[COL_POS, COL_NEG],
        edgecolor="white", width=0.4
    )
    for bar, val in zip(bars, [n_pos, n_neg]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f"{val:,}", ha="center", fontweight="bold", fontsize=12)
    ax.set_ylabel("Number of epitopes")
    ax.set_title(f"Imbalance ratio: {ratio:.1f}:1 (neg:pos)")

    ax.annotate(
        f"Naive model accuracy = {100*n_neg/(n_pos+n_neg):.1f}%\n(predicting all negative)\nThis is misleading!",
        xy=(1, n_neg * 0.7), xytext=(0.5, n_neg * 0.85),
        fontsize=9, color="#B22222",
        arrowprops=dict(arrowstyle="->", color="#B22222", lw=0.8),
    )

    # ── Right: Solutions we will apply ──
    ax2 = axes[1]
    ax2.axis("off")
    solutions_text = (
        "Strategies to handle imbalance\n"
        "─────────────────────────────────────\n\n"
        f"Problem: {ratio:.1f}x more negatives than positives\n\n"
        "Solution 1 — Weighted loss function\n"
        "  GNN assigns higher penalty for missing\n"
        "  a positive (immunogenic) epitope.\n"
        f"  Weight = {ratio:.1f} for positive class.\n\n"
        "Solution 2 — Evaluation metrics\n"
        "  We will NOT use accuracy.\n"
        "  Instead: AUROC, F1, Precision-Recall\n"
        "  These are robust to imbalance.\n\n"
        "Solution 3 — Stratified splits\n"
        "  Train/val/test splits maintain\n"
        "  the same pos:neg ratio in each set."
    )
    ax2.text(0.05, 0.95, solutions_text, transform=ax2.transAxes,
             fontsize=9.5, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF5F5",
                       edgecolor=COL_NEG, linewidth=1))

    plt.tight_layout()
    save(fig, "06_class_imbalance.png")


# ── Plot 7: Position-specific AA heatmap ─────────────────────────────────────

def plot_position_heatmap(data: dict) -> None:
    """
    Why: HLA binding depends heavily on WHICH amino acid is at WHICH POSITION.
    Positions 2 and 9 (for 9-mers) are "anchor positions" — they physically
    slot into pockets on the HLA molecule. This heatmap reveals these patterns.

    For a 9-mer epitope: X-X-X-X-X-X-X-X-X
                                  ↑           ↑
                              position 2  position 9 (anchor positions)

    This kind of analysis is typically Figure 3 or 4 in vaccine papers.
    """
    logger.info("Plot 7: Position-specific amino acid heatmap (9-mers)")

    # Focus on 9-mers — the most common MHC Class I epitopes
    df_pos_9 = data["pos"][data["pos"]["seq_length"] == 9]
    df_neg_9 = data["neg"][data["neg"]["seq_length"] == 9]

    if len(df_pos_9) < 10:
        logger.warning(f"  Only {len(df_pos_9)} 9-mers in positive set — skipping heatmap")
        return

    AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
    n_pos = len(df_pos_9)

    def position_matrix(seqs: pd.Series) -> pd.DataFrame:
        """Build a 20 x 9 frequency matrix: rows=AA, cols=position."""
        n = len(seqs)
        mat = np.zeros((20, 9))
        for seq in seqs:
            seq = str(seq).upper()
            if len(seq) != 9:
                continue
            for pos, aa in enumerate(seq):
                if aa in AA_ORDER:
                    mat[AA_ORDER.index(aa), pos] += 1
        return pd.DataFrame(mat / max(n, 1) * 100,
                            index=AA_ORDER,
                            columns=[f"P{i+1}" for i in range(9)])

    mat_pos = position_matrix(df_pos_9["epitope_seq"])
    mat_neg = position_matrix(df_neg_9["epitope_seq"])
    mat_diff = mat_pos - mat_neg   # enrichment: positive vs negative

    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    fig.suptitle(
        "Position-specific Amino Acid Frequency in 9-mer Epitopes\n"
        f"(MHC Class I, n_pos={n_pos}, n_neg={len(df_neg_9)})",
        fontsize=13, fontweight="bold"
    )

    cmap_freq = "Blues"
    cmap_diff = "RdBu_r"

    # Positive heatmap
    sns.heatmap(mat_pos, ax=axes[0], cmap=cmap_freq, linewidths=0.3,
                cbar_kws={"label": "Freq (%)"}, vmin=0)
    axes[0].set_title("Positive (immunogenic)", fontweight="bold")
    axes[0].set_xlabel("Position in epitope")
    axes[0].set_ylabel("Amino acid")

    # Negative heatmap
    sns.heatmap(mat_neg, ax=axes[1], cmap=cmap_freq, linewidths=0.3,
                cbar_kws={"label": "Freq (%)"}, vmin=0)
    axes[1].set_title("Negative", fontweight="bold")
    axes[1].set_xlabel("Position in epitope")
    axes[1].set_ylabel("")

    # Difference heatmap (the interesting one)
    max_diff = max(abs(mat_diff.values.max()), abs(mat_diff.values.min()))
    sns.heatmap(mat_diff, ax=axes[2], cmap=cmap_diff, linewidths=0.3,
                cbar_kws={"label": "Δ freq (pos − neg)"},
                vmin=-max_diff, vmax=max_diff, center=0)
    axes[2].set_title("Enrichment in immunogenic\n(blue=enriched, red=depleted)",
                      fontweight="bold")
    axes[2].set_xlabel("Position in epitope")
    axes[2].set_ylabel("")

    # Highlight anchor positions P2 and P9
    for ax in axes:
        ax.add_patch(plt.Rectangle((1, 0), 1, 20, fill=False,
                     edgecolor="gold", linewidth=2.5, zorder=5))
        ax.add_patch(plt.Rectangle((8, 0), 1, 20, fill=False,
                     edgecolor="gold", linewidth=2.5, zorder=5))

    axes[2].text(1.5, -0.5, "P2\nanchor", ha="center", fontsize=8,
                 color="goldenrod", fontweight="bold", transform=axes[2].get_xaxis_transform())
    axes[2].text(8.5, -0.5, "P9\nanchor", ha="center", fontsize=8,
                 color="goldenrod", fontweight="bold", transform=axes[2].get_xaxis_transform())

    plt.tight_layout()
    save(fig, "07_position_heatmap_9mer.png")


# ── Summary stats table ───────────────────────────────────────────────────────

def print_eda_summary(data: dict, overlap: set) -> None:
    console.rule("[bold green]EDA Summary Statistics[/bold green]")

    t = Table(title="Dataset Statistics", header_style="bold cyan", show_lines=True)
    t.add_column("Metric",    style="white",       min_width=40)
    t.add_column("Value",     style="bold yellow",  min_width=20)
    t.add_column("Note",      style="dim",           min_width=30)

    n_pos  = len(data["pos"])
    n_neg  = len(data["neg"])
    n_vjdb = len(data["vjdb"])

    t.add_row("Positive (immunogenic) TB epitopes", f"{n_pos:,}",
              "Confirmed T-cell activators")
    t.add_row("Negative (non-immunogenic) epitopes", f"{n_neg:,}",
              "Confirmed non-activators")
    t.add_row("Class imbalance ratio",  f"{n_neg/n_pos:.1f}:1",
              "Must use weighted loss in GNN")
    t.add_row("TCR-confirmed epitopes (VDJdb)", f"{n_vjdb:,}",
              "61 TCR-epitope pairs, 11 unique epitopes")
    t.add_row("Gold standard (IEDB ∩ VDJdb)", f"{len(overlap):,}",
              "Dual-evidence: strongest graph edges")
    t.add_row("HLA alleles in dataset", "44,398",
              "A, B, C, DR, DQ, DP covered")
    t.add_row("TB source proteins available", f"{len(data['tb']):,}",
              "Full H37Rv proteome")
    t.add_row("Avg epitope length (positive)",
              f"{data['pos']['seq_length'].mean():.1f} aa",
              f"Range: {data['pos']['seq_length'].min()}–{data['pos']['seq_length'].max()}")
    t.add_row("Avg epitope length (negative)",
              f"{data['neg']['seq_length'].mean():.1f} aa",
              f"Range: {data['neg']['seq_length'].min()}–{data['neg']['seq_length'].max()}")

    console.print(t)
    console.print(f"\n[bold]Figures saved to:[/bold] {FIGURES_DIR.relative_to(PROJECT_ROOT)}\n")
    console.print("[bold cyan]Next step:[/bold cyan] uv run python scripts/03_feature_engineering.py\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    console.rule("[bold cyan]Phase 2: Exploratory Data Analysis[/bold cyan]")
    console.print("\n[bold]Project:[/bold] GNN-Guided Multi-Epitope Vaccine Design\n")

    data = load_data()

    console.rule("[yellow]Generating figures[/yellow]")
    plot_length_distribution(data)
    plot_source_proteins(data)
    overlap = plot_overlap(data)
    plot_hla_coverage(data)
    plot_amino_acid_frequency(data)
    plot_class_imbalance(data)
    plot_position_heatmap(data)

    print_eda_summary(data, overlap)


if __name__ == "__main__":
    main()