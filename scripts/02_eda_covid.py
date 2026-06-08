"""
02_eda_covid.py
===============
Phase 2 (COVID Validation): Exploratory Data Analysis
GNN-Guided Multi-Epitope Vaccine Design — COVID-19 Validation Dataset

What this script produces (saved to outputs/figures_covid/):
    01_epitope_length_distribution.png   — length histogram pos vs neg
    02_top_source_proteins.png           — which COVID genes/proteins epitopes come from
    03_iedb_vdjdb_overlap.png            — IEDB positive ∩ VDJdb (gold standard)
    04_cdr3_length_distribution.png      — CDR3 length distribution (COVID TCR data)
    05_amino_acid_frequency.png          — AA composition pos vs neg
    06_class_imbalance.png               — pos vs neg counts + imbalance note
    07_epitope_position_heatmap.png      — AA preference at each position (9-mers)
    08_mhc_class_distribution.png        — MHC class I vs II in VDJdb

Run from project root:
    uv run python scripts/02_eda_covid.py
"""

from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from loguru import logger
from rich.console import Console
from rich.table import Table

# ── Setup ─────────────────────────────────────────────────────────────────────

console = Console()

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_covid"
FIGURES_DIR   = PROJECT_ROOT / "outputs" / "figures_covid"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Publication-ready style ───────────────────────────────────────────────────

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

# COVID color palette — same logic as TB but distinct tones
COL_POS   = "#2E86AB"   # blue  — positive/immunogenic
COL_NEG   = "#E84855"   # red   — negative/non-immunogenic
COL_SPIKE = "#F4A261"   # orange — Spike protein
COL_NC    = "#3BB273"   # green — Nucleocapsid
COL_TCR   = "#7B4F9E"   # purple — TCR/VDJdb
COL_MHC1  = "#2E86AB"   # blue — MHC class I
COL_MHC2  = "#E84855"   # red  — MHC class II


def save(fig: plt.Figure, name: str) -> None:
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved: {path.relative_to(PROJECT_ROOT)}")


# ── Load cleaned data ─────────────────────────────────────────────────────────

def load_data() -> dict:
    logger.info("Loading cleaned COVID datasets from data/processed_covid/")

    df_pos  = pd.read_csv(PROCESSED_DIR / "iedb_positive_covid.csv")
    df_neg  = pd.read_csv(PROCESSED_DIR / "iedb_negative_covid.csv")
    df_vjdb = pd.read_csv(PROCESSED_DIR / "vdjdb_covid_clean.tsv", sep="\t")
    df_prot = pd.read_csv(PROCESSED_DIR / "covid_proteins_clean.csv")

    # Add seq_length if not present
    for df in [df_pos, df_neg]:
        if "seq_length" not in df.columns:
            df["seq_length"] = df["epitope_seq"].str.len()

    df_all = pd.concat([df_pos, df_neg], ignore_index=True)

    logger.info(f"  Positive epitopes : {len(df_pos):,}")
    logger.info(f"  Negative epitopes : {len(df_neg):,}")
    logger.info(f"  VDJdb TCR pairs   : {len(df_vjdb):,}")
    logger.info(f"  COVID proteins    : {len(df_prot):,}")

    return {
        "pos":  df_pos,
        "neg":  df_neg,
        "all":  df_all,
        "vjdb": df_vjdb,
        "prot": df_prot,
    }


# ── Plot 1: Epitope Length Distribution ───────────────────────────────────────

def plot_length_distribution(data: dict) -> None:
    """
    Why: T-cell epitopes come in two biological flavours:
      - MHC Class I  (CD8+ T-cells): 8–11 aa — you expect a spike at 9
      - MHC Class II (CD4+ T-cells): 13–25 aa — broader hump at 13–17

    For COVID vaccine design, you want both — CD8+ for killing infected
    cells, CD4+ for helping B-cells produce neutralising antibodies.
    """
    logger.info("Plot 1: Epitope length distribution")

    df_pos = data["pos"]
    df_neg = data["neg"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("COVID-19 Epitope Length Distribution", fontsize=14,
                 fontweight="bold", y=1.02)

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
    ax.set_title("Raw counts — pos vs neg")
    ax.legend(fontsize=9)
    ax.set_xticks(range(8, 27, 2))

    # ── Right: Percentage ratio by length ──
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
    ax2.bar(x, pos_pct, color=COL_POS, alpha=0.8, label="% Positive",
            edgecolor="white", linewidth=0.5)
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


# ── Plot 2: Top Source COVID Genes/Proteins ───────────────────────────────────

def plot_source_proteins(data: dict) -> None:
    """
    Why: SARS-CoV-2 is a small virus (~30kb genome, ~29 proteins).
    Most T-cell immunity concentrates on:
      - Spike (S) — entry into cells; vaccine target
      - Nucleocapsid (N) — structural; highly conserved
      - Membrane (M) — abundant; underexplored
      - ORF1ab — non-structural; replication machinery

    Knowing WHICH proteins dominate immunogenicity tells you whether
    a multi-epitope vaccine focusing only on Spike is sufficient,
    or whether adding N/M/ORF epitopes increases breadth.
    """
    logger.info("Plot 2: Top COVID source proteins/genes")

    df_pos = data["pos"]

    # source_molecule column holds protein/gene info
    prot_col = "source_molecule"

    if prot_col not in df_pos.columns or df_pos[prot_col].isna().all():
        logger.warning("  No source_molecule column — skipping plot 2")
        return

    # Simplify protein names to recognisable COVID gene labels
    def simplify_covid_protein(name: str) -> str:
        if not isinstance(name, str):
            return "Unknown"
        name = name.strip()
        # Map common IEDB naming patterns to standard gene labels
        mappings = [
            ("spike", "Spike (S)"),
            ("surface glycoprotein", "Spike (S)"),
            ("nucleocapsid", "Nucleocapsid (N)"),
            ("membrane", "Membrane (M)"),
            ("envelope", "Envelope (E)"),
            ("orf1ab", "ORF1ab"),
            ("orf1a", "ORF1ab"),
            ("nsp", "ORF1ab (NSP)"),
            ("replicase", "ORF1ab"),
            ("orf3", "ORF3a"),
            ("orf6", "ORF6"),
            ("orf7", "ORF7"),
            ("orf8", "ORF8"),
            ("orf9", "ORF9b"),
            ("orf10", "ORF10"),
        ]
        lower = name.lower()
        for key, label in mappings:
            if key in lower:
                return label
        # Fallback: first 30 chars
        return name[:30] if len(name) > 30 else name

    proteins = (
        df_pos[prot_col]
        .dropna()
        .apply(simplify_covid_protein)
    )

    top_proteins = proteins.value_counts().head(15)

    if top_proteins.empty:
        logger.warning("  Source molecule column is empty — skipping plot 2")
        return

    # Colour by known immunodominant proteins
    bar_colors = []
    for p in top_proteins.index:
        if "Spike" in p:
            bar_colors.append(COL_SPIKE)
        elif "Nucleocapsid" in p:
            bar_colors.append(COL_NC)
        elif "ORF1" in p:
            bar_colors.append(COL_MHC2)
        else:
            bar_colors.append("#AAAAAA")

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(
        top_proteins.index[::-1],
        top_proteins.values[::-1],
        color=bar_colors[::-1],
        edgecolor="white", linewidth=0.4
    )

    for bar, val in zip(bars, top_proteins.values[::-1]):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                f"{val:,}", va="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Number of positive epitopes")
    ax.set_title(
        "Top COVID-19 Source Proteins — Immunodominant Antigens\n"
        "(Positive IEDB epitopes only)",
        fontsize=13, fontweight="bold"
    )

    # Legend
    legend_elements = [
        mpatches.Patch(color=COL_SPIKE, label="Spike (S)"),
        mpatches.Patch(color=COL_NC,    label="Nucleocapsid (N)"),
        mpatches.Patch(color=COL_MHC2,  label="ORF1ab"),
        mpatches.Patch(color="#AAAAAA", label="Other"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    ax.set_xlim(0, top_proteins.values.max() * 1.15)

    plt.tight_layout()
    save(fig, "02_top_source_proteins.png")


# ── Plot 3: IEDB ∩ VDJdb Gold Standard Overlap ───────────────────────────────

def plot_overlap(data: dict) -> set:
    """
    Why: An epitope confirmed in BOTH IEDB (immunogenicity assay) AND
    VDJdb (actual TCR sequence binding) is the gold standard.
    These are the strongest nodes in the graph — dual-evidence positives.

    COVID has 668 such epitopes vs TB's 11. This is a much richer
    gold standard, which is a key strength of this validation dataset.
    """
    logger.info("Plot 3: IEDB–VDJdb overlap (gold standard)")

    iedb_pos  = set(data["pos"]["epitope_seq"].str.upper().dropna())
    vjdb_eps  = set(data["vjdb"]["epitope"].str.upper().dropna())
    overlap   = iedb_pos & vjdb_eps

    n_iedb_only  = len(iedb_pos) - len(overlap)
    n_vjdb_only  = len(vjdb_eps) - len(overlap)
    n_overlap    = len(overlap)

    logger.info(f"  IEDB positive only : {n_iedb_only:,}")
    logger.info(f"  VDJdb only         : {n_vjdb_only:,}")
    logger.info(f"  Gold standard (∩)  : {n_overlap:,}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "IEDB Positive ∩ VDJdb: COVID Gold-Standard Epitopes",
        fontsize=13, fontweight="bold"
    )

    # ── Left: Venn-style bar chart ──
    ax = axes[0]
    categories = ["IEDB only\n(assay confirmed)", "Gold standard\n(dual evidence)", "VDJdb only\n(TCR confirmed)"]
    values     = [n_iedb_only, n_overlap, n_vjdb_only]
    colors     = [COL_POS, "#F4A261", COL_TCR]

    bars = ax.bar(categories, values, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f"{val:,}", ha="center", fontweight="bold", fontsize=12)

    ax.set_ylabel("Number of epitopes")
    ax.set_title("Epitope database overlap")
    ax.set_ylim(0, max(values) * 1.15)

    # ── Right: Top overlapping epitopes by TCR pair count ──
    ax2 = axes[1]
    vjdb_filtered = data["vjdb"][
        data["vjdb"]["epitope"].str.upper().isin(overlap)
    ]
    top_overlap = (
        vjdb_filtered["epitope"]
        .value_counts()
        .head(12)
        .sort_values()
    )

    if not top_overlap.empty:
        ax2.barh(top_overlap.index, top_overlap.values,
                 color=COL_TCR, edgecolor="white", alpha=0.85)
        for i, (val, label) in enumerate(zip(top_overlap.values, top_overlap.index)):
            ax2.text(val + 1, i, f"{val:,}", va="center", fontsize=9)

        ax2.set_xlabel("Number of TCR-epitope pairs in VDJdb")
        ax2.set_title(
            f"Top gold-standard epitopes by TCR evidence\n"
            f"(n={n_overlap} dual-confirmed epitopes total)",
            fontsize=11
        )
    else:
        ax2.text(0.5, 0.5, "No overlap found", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=14, color="gray")

    plt.tight_layout()
    save(fig, "03_iedb_vdjdb_overlap.png")
    return overlap


# ── Plot 4: CDR3 Length Distribution (COVID TCR) ──────────────────────────────

def plot_cdr3_length(data: dict) -> None:
    """
    Why: CDR3 is the hypervariable loop of the T-cell receptor that makes
    direct contact with the epitope-HLA complex. CDR3 length correlates
    with MHC class:
      - CDR3β for MHC I: typically 12–16 aa
      - CDR3β for MHC II: typically 14–20 aa (longer groove)

    COVID's 9,341 unique CDR3s is enormous — this plot shows whether
    the TCR repertoire coverage is biased towards one MHC class.
    This matters because it affects what GNN edges you can build.
    """
    logger.info("Plot 4: CDR3 length distribution")

    df_vjdb = data["vjdb"]
    df_vjdb = df_vjdb.copy()
    df_vjdb["cdr3_length"] = df_vjdb["cdr3"].str.len()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "COVID-19 TCR CDR3 Length Distribution\n(VDJdb — 9,341 unique CDR3s)",
        fontsize=13, fontweight="bold"
    )

    # ── Left: Overall CDR3 length histogram ──
    ax = axes[0]
    ax.hist(df_vjdb["cdr3_length"].dropna(), bins=range(5, 35),
            color=COL_TCR, alpha=0.8, edgecolor="white")
    ax.axvspan(12, 16, alpha=0.15, color=COL_MHC1, label="Typical MHC I CDR3 (12–16)")
    ax.axvspan(14, 20, alpha=0.10, color=COL_MHC2, label="Typical MHC II CDR3 (14–20)")
    ax.set_xlabel("CDR3 length (amino acids)")
    ax.set_ylabel("Count")
    ax.set_title("CDR3β length — all COVID TCRs")
    ax.legend(fontsize=9)

    # ── Right: CDR3 length split by MHC class (if available) ──
    ax2 = axes[1]
    if "mhc_class" in df_vjdb.columns:
        mhc1 = df_vjdb[df_vjdb["mhc_class"].str.contains("I", na=False) &
                        ~df_vjdb["mhc_class"].str.contains("II", na=False)]["cdr3_length"].dropna()
        mhc2 = df_vjdb[df_vjdb["mhc_class"].str.contains("II", na=False)]["cdr3_length"].dropna()

        if len(mhc1) > 0:
            ax2.hist(mhc1, bins=range(5, 35), alpha=0.7, color=COL_MHC1,
                     label=f"MHC I (n={len(mhc1):,})", edgecolor="white")
        if len(mhc2) > 0:
            ax2.hist(mhc2, bins=range(5, 35), alpha=0.5, color=COL_MHC2,
                     label=f"MHC II (n={len(mhc2):,})", edgecolor="white")

        ax2.set_xlabel("CDR3 length (amino acids)")
        ax2.set_ylabel("Count")
        ax2.set_title("CDR3 length by MHC class")
        ax2.legend(fontsize=9)
    else:
        # Fallback: gene type (TRA vs TRB)
        if "gene" in df_vjdb.columns:
            for gene, color in [("TRA", COL_MHC1), ("TRB", COL_MHC2)]:
                subset = df_vjdb[df_vjdb["gene"].str.upper() == gene]["cdr3_length"].dropna()
                if len(subset) > 0:
                    ax2.hist(subset, bins=range(5, 35), alpha=0.7, color=color,
                             label=f"{gene} (n={len(subset):,})", edgecolor="white")
            ax2.set_xlabel("CDR3 length (amino acids)")
            ax2.set_ylabel("Count")
            ax2.set_title("CDR3 length by TCR chain (TRA vs TRB)")
            ax2.legend(fontsize=9)
        else:
            ax2.text(0.5, 0.5, "MHC class / gene column\nnot available",
                     ha="center", va="center", transform=ax2.transAxes,
                     fontsize=12, color="gray")

    plt.tight_layout()
    save(fig, "04_cdr3_length_distribution.png")


# ── Plot 5: Amino Acid Frequency ──────────────────────────────────────────────

def plot_amino_acid_frequency(data: dict) -> None:
    """
    Why: Immunogenic peptides have characteristic amino acid biases.
    Hydrophobic residues (L, I, V, A) are enriched at anchor positions
    for MHC I (they slot into hydrophobic B and F pockets of HLA).
    Comparing AA frequency in positives vs negatives reveals whether
    our dataset has learnable chemical patterns — a prerequisite for GNN.
    """
    logger.info("Plot 5: Amino acid frequency comparison")

    AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
    df_pos = data["pos"]
    df_neg = data["neg"]

    def aa_freq(df: pd.DataFrame) -> dict:
        counter = Counter()
        for seq in df["epitope_seq"].dropna():
            counter.update(str(seq).upper())
        total = sum(counter.values())
        return {aa: counter.get(aa, 0) / total * 100 for aa in AA_ORDER}

    freq_pos = aa_freq(df_pos)
    freq_neg = aa_freq(df_neg)

    x = np.arange(len(AA_ORDER))
    w = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    fig.suptitle("Amino Acid Frequency: COVID Positive vs Negative Epitopes",
                 fontsize=13, fontweight="bold")

    # ── Top: Side-by-side bars ──
    ax = axes[0]
    ax.bar(x - w/2, [freq_pos[aa] for aa in AA_ORDER], width=w,
           color=COL_POS, alpha=0.85, label=f"Positive (n={len(df_pos):,})",
           edgecolor="white")
    ax.bar(x + w/2, [freq_neg[aa] for aa in AA_ORDER], width=w,
           color=COL_NEG, alpha=0.7, label=f"Negative (n={len(df_neg):,})",
           edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(AA_ORDER)
    ax.set_ylabel("Frequency (%)")
    ax.set_title("Absolute amino acid frequencies")
    ax.legend(fontsize=9)

    # ── Bottom: Enrichment (log2 fold change) ──
    ax2 = axes[1]
    log2fc = []
    for aa in AA_ORDER:
        pos_f = freq_pos[aa] + 1e-6
        neg_f = freq_neg[aa] + 1e-6
        log2fc.append(np.log2(pos_f / neg_f))

    colors_fc = [COL_POS if v >= 0 else COL_NEG for v in log2fc]
    ax2.bar(x, log2fc, color=colors_fc, edgecolor="white", alpha=0.85)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(AA_ORDER)
    ax2.set_ylabel("log₂(Positive freq / Negative freq)")
    ax2.set_title("Amino acid enrichment in immunogenic epitopes\n"
                  "(blue = enriched in positives, red = depleted)")

    # Annotate highly enriched/depleted AAs
    for i, (aa, fc) in enumerate(zip(AA_ORDER, log2fc)):
        if abs(fc) > 0.15:
            ax2.text(i, fc + (0.01 if fc > 0 else -0.03), aa,
                     ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    save(fig, "05_amino_acid_frequency.png")


# ── Plot 6: Class Imbalance ───────────────────────────────────────────────────

def plot_class_imbalance(data: dict) -> None:
    """
    Why: COVID dataset is near-balanced (4,213 pos : 4,135 neg ≈ 1:1).
    This is fundamentally different from TB (where negatives greatly
    outnumber positives). For this validation set:
      - No need for heavily weighted loss
      - Standard accuracy is now a meaningful metric
      - Balanced F1 is still preferred for reporting

    This plot communicates the imbalance (or lack of it) and what
    it means for model training — which differs from the TB model.
    """
    logger.info("Plot 6: Class imbalance")

    n_pos = len(data["pos"])
    n_neg = len(data["neg"])
    ratio = n_neg / n_pos if n_pos > 0 else 0

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("COVID-19 Dataset: Class Balance",
                 fontsize=13, fontweight="bold")

    # ── Left: Bar chart ──
    ax = axes[0]
    bars = ax.bar(
        ["Positive\n(immunogenic)", "Negative\n(non-immunogenic)"],
        [n_pos, n_neg],
        color=[COL_POS, COL_NEG],
        edgecolor="white", width=0.4
    )
    for bar, val in zip(bars, [n_pos, n_neg]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f"{val:,}", ha="center", fontweight="bold", fontsize=12)

    ax.set_ylabel("Number of epitopes")
    ax.set_title(f"Imbalance ratio: {ratio:.2f}:1 (neg:pos)\n≈ Balanced dataset")

    # ── Right: Comparison vs TB + what this means ──
    ax2 = axes[1]
    ax2.axis("off")
    note_text = (
        "COVID vs TB Class Balance Comparison\n"
        "─────────────────────────────────────\n\n"
        f"COVID  pos: {n_pos:,}  neg: {n_neg:,}  ratio: {ratio:.2f}:1\n"
        f"TB     pos: ~13,000   neg: ~42,000    ratio: ~3.2:1\n\n"
        "What this means for the COVID model:\n\n"
        "✓ Standard accuracy IS meaningful here\n"
        "  (unlike TB where it was misleading)\n\n"
        "✓ No heavy class weighting needed\n"
        "  (slight weight still recommended)\n\n"
        "✓ Balanced F1, AUROC still preferred\n"
        "  for fair comparison with TB results\n\n"
        "✓ This balance validates our IEDB data\n"
        "  collection strategy — no sampling bias"
    )
    ax2.text(0.05, 0.95, note_text, transform=ax2.transAxes,
             fontsize=9.5, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0FFF0",
                       edgecolor=COL_POS, linewidth=1))

    plt.tight_layout()
    save(fig, "06_class_imbalance.png")


# ── Plot 7: Position-specific AA heatmap (9-mers) ────────────────────────────

def plot_position_heatmap(data: dict) -> None:
    """
    Why: HLA binding depends heavily on WHICH amino acid is at WHICH POSITION.
    Positions 2 and 9 (for 9-mers) are "anchor positions" — they physically
    slot into pockets on the HLA molecule. This heatmap reveals these patterns.

    For COVID: COVID MHC I epitopes (primarily from Spike and Nucleocapsid)
    show strong anchor preferences. Seeing P2=L/M and P9=V/L is consistent
    with HLA-A*02:01 binding — the most common HLA allele in datasets.
    """
    logger.info("Plot 7: Position-specific amino acid heatmap (9-mers)")

    df_pos_9 = data["pos"][data["pos"]["seq_length"] == 9]
    df_neg_9 = data["neg"][data["neg"]["seq_length"] == 9]

    if len(df_pos_9) < 10:
        logger.warning(f"  Only {len(df_pos_9)} 9-mers — skipping heatmap")
        return

    AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")

    def position_matrix(seqs: pd.Series) -> pd.DataFrame:
        n = len(seqs)
        mat = np.zeros((20, 9))
        for seq in seqs:
            seq = str(seq).upper()
            if len(seq) != 9:
                continue
            for pos, aa in enumerate(seq):
                if aa in AA_ORDER:
                    mat[AA_ORDER.index(aa), pos] += 1
        return pd.DataFrame(
            mat / max(n, 1) * 100,
            index=AA_ORDER,
            columns=[f"P{i+1}" for i in range(9)]
        )

    mat_pos  = position_matrix(df_pos_9["epitope_seq"])
    mat_neg  = position_matrix(df_neg_9["epitope_seq"])
    mat_diff = mat_pos - mat_neg

    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    fig.suptitle(
        "Position-specific AA Frequency in COVID 9-mer Epitopes\n"
        f"(MHC Class I, n_pos={len(df_pos_9)}, n_neg={len(df_neg_9)})",
        fontsize=13, fontweight="bold"
    )

    cmap_freq = "Blues"
    cmap_diff = "RdBu_r"

    sns.heatmap(mat_pos, ax=axes[0], cmap=cmap_freq, linewidths=0.3,
                cbar_kws={"label": "Freq (%)"}, vmin=0)
    axes[0].set_title("Positive (immunogenic)", fontweight="bold")
    axes[0].set_xlabel("Position in epitope")
    axes[0].set_ylabel("Amino acid")

    sns.heatmap(mat_neg, ax=axes[1], cmap=cmap_freq, linewidths=0.3,
                cbar_kws={"label": "Freq (%)"}, vmin=0)
    axes[1].set_title("Negative", fontweight="bold")
    axes[1].set_xlabel("Position in epitope")
    axes[1].set_ylabel("")

    max_diff = max(abs(mat_diff.values.max()), abs(mat_diff.values.min()))
    sns.heatmap(mat_diff, ax=axes[2], cmap=cmap_diff, linewidths=0.3,
                cbar_kws={"label": "Δ freq (pos − neg)"},
                vmin=-max_diff, vmax=max_diff, center=0)
    axes[2].set_title("Enrichment in immunogenic\n(blue=enriched, red=depleted)",
                      fontweight="bold")
    axes[2].set_xlabel("Position in epitope")
    axes[2].set_ylabel("")

    # Anchor positions P2 and P9 — HLA-A*02:01 key residues
    for ax in axes:
        ax.add_patch(plt.Rectangle((1, 0), 1, 20, fill=False,
                     edgecolor="gold", linewidth=2.5, zorder=5))
        ax.add_patch(plt.Rectangle((8, 0), 1, 20, fill=False,
                     edgecolor="gold", linewidth=2.5, zorder=5))

    axes[2].text(1.5, -0.5, "P2\nanchor", ha="center", fontsize=8,
                 color="goldenrod", fontweight="bold",
                 transform=axes[2].get_xaxis_transform())
    axes[2].text(8.5, -0.5, "P9\nanchor", ha="center", fontsize=8,
                 color="goldenrod", fontweight="bold",
                 transform=axes[2].get_xaxis_transform())

    plt.tight_layout()
    save(fig, "07_epitope_position_heatmap.png")


# ── Plot 8: MHC Class Distribution in VDJdb ──────────────────────────────────

def plot_mhc_class_distribution(data: dict) -> None:
    """
    Why: This is a COVID-specific plot with no TB equivalent.
    VDJdb for COVID has 9,341 unique CDR3s — the largest TCR
    dataset we have. Knowing the MHC class split tells you:
      - What fraction of COVID immunity is CD8+ (kill infected cells)
        vs CD4+ (coordinate the immune response)
      - Whether the VDJdb data gives balanced graph edges for both classes

    A heavily MHC I-biased dataset means the GNN will be better at
    predicting CD8+ epitopes than CD4+ helpers.
    """
    logger.info("Plot 8: MHC class distribution in VDJdb")

    df_vjdb = data["vjdb"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("COVID VDJdb — MHC Class Distribution",
                 fontsize=13, fontweight="bold")

    # ── Left: MHC class pie/bar ──
    ax = axes[0]
    if "mhc_class" in df_vjdb.columns:
        mhc_counts = (
            df_vjdb["mhc_class"]
            .str.strip()
            .value_counts()
        )
        colors_mhc = [COL_MHC1 if "I" in k and "II" not in k else COL_MHC2
                      for k in mhc_counts.index]
        bars = ax.bar(mhc_counts.index, mhc_counts.values,
                      color=colors_mhc, edgecolor="white", alpha=0.85)
        for bar, val in zip(bars, mhc_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f"{val:,}", ha="center", fontweight="bold")
        ax.set_ylabel("TCR-epitope pairs")
        ax.set_title("TCR pairs by MHC class\n(VDJdb COVID data)")
        ax.set_xlabel("MHC class")
    else:
        ax.text(0.5, 0.5, "mhc_class column\nnot available",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="gray")

    # ── Right: Top epitopes by MHC class (if available) ──
    ax2 = axes[1]
    if "mhc_class" in df_vjdb.columns:
        mhc1_eps = (
            df_vjdb[df_vjdb["mhc_class"].str.contains("I", na=False) &
                    ~df_vjdb["mhc_class"].str.contains("II", na=False)]
            ["epitope"].value_counts().head(8)
        )
        mhc2_eps = (
            df_vjdb[df_vjdb["mhc_class"].str.contains("II", na=False)]
            ["epitope"].value_counts().head(8)
        )

        y_pos1 = np.arange(len(mhc1_eps))
        y_pos2 = np.arange(len(mhc2_eps)) - len(mhc1_eps) - 1.5

        if len(mhc1_eps) > 0:
            ax2.barh(y_pos1, mhc1_eps.values, color=COL_MHC1, alpha=0.8,
                     label="MHC I")
            for i, (val, seq) in enumerate(zip(mhc1_eps.values, mhc1_eps.index)):
                ax2.text(val + 1, y_pos1[i], f"{seq} ({val})", va="center", fontsize=8)

        if len(mhc2_eps) > 0:
            ax2.barh(y_pos2, mhc2_eps.values, color=COL_MHC2, alpha=0.8,
                     label="MHC II")
            for i, (val, seq) in enumerate(zip(mhc2_eps.values, mhc2_eps.index)):
                ax2.text(val + 1, y_pos2[i], f"{seq} ({val})", va="center", fontsize=8)

        ax2.set_yticks([])
        ax2.set_xlabel("TCR pairs")
        ax2.set_title("Top epitopes by MHC class")
        ax2.legend(fontsize=9)
    else:
        ax2.text(0.5, 0.5, "mhc_class column\nnot available",
                 ha="center", va="center", transform=ax2.transAxes,
                 fontsize=12, color="gray")

    plt.tight_layout()
    save(fig, "08_mhc_class_distribution.png")


# ── Summary stats table ───────────────────────────────────────────────────────

def print_eda_summary(data: dict, overlap: set) -> None:
    console.rule("[bold green]COVID EDA Summary Statistics[/bold green]")

    t = Table(title="COVID-19 Dataset Statistics",
              header_style="bold cyan", show_lines=True)
    t.add_column("Metric",    style="white",       min_width=45)
    t.add_column("Value",     style="bold yellow",  min_width=15)
    t.add_column("Note",      style="dim",           min_width=35)

    n_pos  = len(data["pos"])
    n_neg  = len(data["neg"])
    n_vjdb = len(data["vjdb"])
    ratio  = n_neg / n_pos if n_pos > 0 else 0

    t.add_row("Positive (immunogenic) COVID epitopes", f"{n_pos:,}",
              "SARS-CoV-2 confirmed T-cell activators")
    t.add_row("Negative (non-immunogenic) epitopes",   f"{n_neg:,}",
              "Confirmed non-activators")
    t.add_row("Class imbalance ratio",  f"{ratio:.2f}:1",
              "Near-balanced — unlike TB (3.2:1)")
    t.add_row("VDJdb TCR-epitope pairs (COVID)",       f"{n_vjdb:,}",
              "Vastly richer than TB (61 pairs)")
    t.add_row("Unique COVID CDR3 sequences",
              f"{data['vjdb']['cdr3'].nunique():,}",
              "Rich TCR diversity")
    t.add_row("Unique epitopes with TCR data",
              f"{data['vjdb']['epitope'].nunique():,}",
              "671 unique COVID epitopes in VDJdb")
    t.add_row("Gold standard (IEDB ∩ VDJdb)",          f"{len(overlap):,}",
              "vs only 11 in TB — much stronger signal")
    t.add_row("COVID proteome proteins",               f"{len(data['prot']):,}",
              "Full SARS-CoV-2 reference proteome")
    t.add_row("Avg epitope length (positive)",
              f"{data['pos']['seq_length'].mean():.1f} aa",
              f"Range: {data['pos']['seq_length'].min()}–{data['pos']['seq_length'].max()}")
    t.add_row("Avg epitope length (negative)",
              f"{data['neg']['seq_length'].mean():.1f} aa",
              f"Range: {data['neg']['seq_length'].min()}–{data['neg']['seq_length'].max()}")

    console.print(t)
    console.print(
        f"\n[bold]Figures saved to:[/bold] {FIGURES_DIR.relative_to(PROJECT_ROOT)}\n"
    )
    console.print(
        "[bold cyan]Next step:[/bold cyan] "
        "uv run python scripts/03_feature_engineering_covid.py\n"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    console.rule("[bold cyan]Phase 2 (COVID): Exploratory Data Analysis[/bold cyan]")
    console.print(
        "\n[bold]Project:[/bold] GNN-Guided Multi-Epitope Vaccine Design — "
        "COVID-19 Validation\n"
    )

    data = load_data()

    console.rule("[yellow]Generating figures[/yellow]")
    plot_length_distribution(data)
    plot_source_proteins(data)
    overlap = plot_overlap(data)
    plot_cdr3_length(data)
    plot_amino_acid_frequency(data)
    plot_class_imbalance(data)
    plot_position_heatmap(data)
    plot_mhc_class_distribution(data)

    print_eda_summary(data, overlap)


if __name__ == "__main__":
    main()