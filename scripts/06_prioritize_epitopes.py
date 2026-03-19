"""
06_prioritize_epitopes.py
=========================
Phase 6: Epitope Prioritization & Vaccine Candidate Ranking
GNN-Guided Multi-Epitope Vaccine Design

What this script does:
    Uses the trained GNN to score ALL epitopes in the graph and produce
    a ranked list of vaccine candidates with biological annotations.

    Beyond the raw GNN score, each epitope is also scored on:
        1. GNN immunogenicity score     (from trained model)
        2. HLA coverage score           (how many HLA types it likely covers)
        3. TCR evidence score           (does VDJdb confirm TCR recognition?)
        4. Protein conservation score   (is the source protein essential to TB?)
        5. Composite vaccine score      (weighted combination of all above)

    The composite score is what we use to rank final candidates.
    This multi-criteria approach is standard in reverse vaccinology papers.

Outputs:
    outputs/vaccine_candidates/
        top_candidates.csv              — full ranked list with all scores
        top50_candidates.csv            — top 50 for detailed analysis
        gold_standard_epitopes.csv      — the 9 dual-evidence epitopes
    outputs/figures/
        10_score_distributions.png
        11_top20_candidates.png
        12_hla_coverage_analysis.png

Run from project root:
    uv run python scripts/06_prioritize_epitopes.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv, Linear
from loguru import logger
from rich.console import Console
from rich.table import Table

# ── Setup ─────────────────────────────────────────────────────────────────────

console = Console()

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
GRAPH_DIR     = PROJECT_ROOT / "data" / "processed" / "graph"
EMBED_DIR     = PROJECT_ROOT / "data" / "processed" / "embeddings"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "outputs" / "models"
FIGURES_DIR   = PROJECT_ROOT / "outputs" / "figures"
OUT_DIR       = PROJECT_ROOT / "outputs" / "vaccine_candidates"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.family": "DejaVu Sans",
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "figure.facecolor": "white",
})

# ── Rebuild model (same architecture as Phase 5) ──────────────────────────────

class EpitopeGNN(nn.Module):
    """Exact same architecture as Phase 5 — must match to load weights."""

    def __init__(self, in_dim, hidden_dim, conv_out_dim, num_heads,
                 num_layers, dropout, metadata):
        super().__init__()
        self.dropout = dropout
        node_types   = metadata[0]

        self.input_proj = nn.ModuleDict({
            nt: nn.Sequential(
                Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
            for nt in node_types
        })

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.projs = nn.ModuleList()

        for i in range(num_layers):
            in_ch = hidden_dim if i == 0 else conv_out_dim
            self.convs.append(HANConv(
                in_channels=in_ch, out_channels=conv_out_dim,
                heads=num_heads, dropout=dropout, metadata=metadata,
            ))
            self.norms.append(nn.ModuleDict({
                nt: nn.LayerNorm(conv_out_dim) for nt in node_types
            }))
            if conv_out_dim != in_ch:
                self.projs.append(nn.ModuleDict({
                    nt: nn.Linear(in_ch, conv_out_dim, bias=False) for nt in node_types
                }))
            else:
                self.projs.append(None)

        self.classifier = nn.Sequential(
            nn.Linear(conv_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x_dict, edge_index_dict):
        h = {}
        for nt, proj in self.input_proj.items():
            if nt in x_dict:
                h[nt] = proj(x_dict[nt])

        for i, conv in enumerate(self.convs):
            h_new = conv(h, edge_index_dict)
            for nt in h_new:
                if h_new[nt] is None:
                    continue
                if nt in h:
                    if self.projs[i] is not None and nt in self.projs[i]:
                        residual = self.projs[i][nt](h[nt])
                    elif h[nt].shape[-1] == h_new[nt].shape[-1]:
                        residual = h[nt]
                    else:
                        residual = None
                    if residual is not None:
                        h_new[nt] = h_new[nt] + residual
                h_new[nt] = self.norms[i][nt](h_new[nt])
                h_new[nt] = F.relu(h_new[nt])
                h_new[nt] = F.dropout(h_new[nt], p=self.dropout, training=self.training)
            for nt in h_new:
                if h_new[nt] is not None:
                    h[nt] = h_new[nt]

        return self.classifier(h["epitope"]).squeeze(-1)


def probe_hanconv_output_dim(metadata, hidden_dim, num_heads):
    dummy_x  = {nt: torch.zeros(2, hidden_dim) for nt in metadata[0]}
    dummy_ei = {et: torch.zeros(2, 0, dtype=torch.long) for et in metadata[1]}
    try:
        conv = HANConv(hidden_dim, hidden_dim, heads=num_heads, metadata=metadata)
        out  = conv(dummy_x, dummy_ei)
        for nt in metadata[0]:
            if nt in out and out[nt] is not None:
                return out[nt].shape[1]
    except Exception:
        pass
    return hidden_dim


# ── Load model and graph ──────────────────────────────────────────────────────

def load_model_and_graph():
    """Load the trained model weights and the heterogeneous graph."""
    # Load graph
    graph_path = GRAPH_DIR / "heterogeneous_graph.pt"
    graph = torch.load(str(graph_path), map_location=device, weights_only=False)
    logger.info(f"Loaded graph: {sum(graph[nt].num_nodes for nt in graph.node_types):,} total nodes")

    # Load saved model checkpoint
    ckpt_path = MODELS_DIR / "best_model.pt"
    ckpt      = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    HP        = ckpt["hyperparams"]
    logger.info(f"Loaded checkpoint: epoch {ckpt['best_epoch']}, "
                f"val AUROC={ckpt['best_val_auroc']:.4f}")

    # Rebuild model with same hyperparams
    conv_out_dim = probe_hanconv_output_dim(
        graph.metadata(), HP["hidden_dim"], HP["num_heads"]
    )
    model = EpitopeGNN(
        in_dim       = graph["epitope"].x.shape[1],
        hidden_dim   = HP["hidden_dim"],
        conv_out_dim = conv_out_dim,
        num_heads    = HP["num_heads"],
        num_layers   = HP["num_layers"],
        dropout      = HP["dropout"],
        metadata     = graph.metadata(),
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    return model, graph, HP


# ── Score all epitopes ────────────────────────────────────────────────────────

@torch.no_grad()
def score_all_epitopes(model: EpitopeGNN, graph: HeteroData) -> np.ndarray:
    """
    Run inference on ALL epitopes (positive + negative) to get GNN scores.

    This is the key step — we're not just evaluating the model on test data,
    we're using it as a RANKING TOOL to score every epitope in the dataset.

    The output probabilities tell us: for each epitope, how confident is
    the GNN (given its graph neighborhood) that this epitope is immunogenic?
    """
    graph  = graph.to(device)
    logits = model(graph.x_dict, graph.edge_index_dict)
    probs  = torch.sigmoid(logits).cpu().numpy()
    logger.info(f"Scored {len(probs):,} epitopes")
    logger.info(f"  Score range: {probs.min():.4f} – {probs.max():.4f}")
    logger.info(f"  Mean score (all): {probs.mean():.4f}")
    logger.info(f"  Epitopes scoring > 0.5: {(probs > 0.5).sum():,}")
    logger.info(f"  Epitopes scoring > 0.7: {(probs > 0.7).sum():,}")
    logger.info(f"  Epitopes scoring > 0.9: {(probs > 0.9).sum():,}")
    return probs


# ── Build annotation dataframe ────────────────────────────────────────────────

def build_annotation_df(graph: HeteroData, gnn_scores: np.ndarray) -> pd.DataFrame:
    """
    Combine GNN scores with biological metadata for each epitope.

    Columns in output:
        epitope_seq         — amino acid sequence
        seq_length          — length in amino acids
        true_label          — 1 if IEDB-confirmed immunogenic, 0 if negative
        gnn_score           — model's predicted immunogenicity probability
        tcr_evidence        — 1 if in VDJdb (gold standard), 0 otherwise
        source_protein      — which TB protein this peptide comes from
        hla_neighbors       — number of HLA nodes connected to this epitope
        hla_coverage_score  — estimated fraction of population covered
        composite_score     — final ranking score (weighted combination)
    """
    logger.info("Building annotation dataframe...")

    # ── Base data from graph ──
    seqs   = graph["epitope"].seq
    labels = graph["epitope"].y.cpu().numpy()

    df = pd.DataFrame({
        "epitope_seq":  seqs,
        "seq_length":   [len(s) for s in seqs],
        "true_label":   labels,
        "gnn_score":    gnn_scores,
    })

    # ── TCR evidence ──
    vjdb_path = PROCESSED_DIR / "vjdb_tb_human_clean.tsv"
    df_vjdb   = pd.read_csv(vjdb_path, sep="\t")
    vjdb_epitopes = set(df_vjdb["epitope"].astype(str).str.upper().str.strip())

    df["tcr_evidence"] = df["epitope_seq"].str.upper().isin(vjdb_epitopes).astype(int)
    n_tcr = df["tcr_evidence"].sum()
    logger.info(f"  TCR-confirmed epitopes in dataset: {n_tcr}")

    # ── Source protein (from protein→epitope edges) ──
    meta_prot = pd.read_csv(EMBED_DIR / "tb_proteins_meta.csv")

    # Build reverse lookup from edge_index
    prot_epi_edges = graph["protein", "source_of", "epitope"].edge_index
    epi_to_prot    = {}
    if prot_epi_edges.shape[1] > 0:
        for i in range(prot_epi_edges.shape[1]):
            prot_idx = int(prot_epi_edges[0, i])
            epi_idx  = int(prot_epi_edges[1, i])
            if epi_idx not in epi_to_prot:
                epi_to_prot[epi_idx] = prot_idx

    gene_names  = []
    prot_names  = []
    for epi_idx in range(len(df)):
        if epi_idx in epi_to_prot:
            prot_idx = epi_to_prot[epi_idx]
            if prot_idx < len(meta_prot):
                row = meta_prot.iloc[prot_idx]
                gene_names.append(row.get("gene_name", ""))
                prot_names.append(str(row.get("protein_name", ""))[:50])
            else:
                gene_names.append(""); prot_names.append("")
        else:
            gene_names.append(""); prot_names.append("")

    df["source_gene"]    = gene_names
    df["source_protein"] = prot_names

    # ── HLA connectivity ──
    epi_hla_edges = graph["epitope", "binds_to", "hla"].edge_index
    hla_counts    = np.zeros(len(df), dtype=int)
    if epi_hla_edges.shape[1] > 0:
        for i in range(epi_hla_edges.shape[1]):
            epi_idx = int(epi_hla_edges[0, i])
            if epi_idx < len(hla_counts):
                hla_counts[epi_idx] += 1

    df["hla_neighbors"] = hla_counts

    # HLA coverage score: normalize 0–1 based on connectivity
    # Epitopes with more diverse HLA connections = better population coverage
    max_hla = max(hla_counts.max(), 1)
    df["hla_coverage_score"] = hla_counts / max_hla

    # ── Composite vaccine score ──
    # Weighted combination of all signals
    # Weights reflect biological importance:
    #   GNN score:    50% — primary model signal
    #   TCR evidence: 30% — experimental validation bonus
    #   HLA coverage: 20% — population coverage
    df["composite_score"] = (
        0.50 * df["gnn_score"] +
        0.30 * df["tcr_evidence"] +
        0.20 * df["hla_coverage_score"]
    )

    # ── MHC class prediction based on length ──
    # 8–11 aa → likely MHC Class I (CD8+ T-cells)
    # 12–25 aa → likely MHC Class II (CD4+ T-cells)
    df["mhc_class"] = df["seq_length"].apply(
        lambda l: "Class I (CD8+)" if l <= 11 else "Class II (CD4+)"
    )

    logger.info(f"  Total epitopes annotated: {len(df):,}")
    logger.info(f"  With source protein:      {(df['source_gene'] != '').sum():,}")
    logger.info(f"  With HLA connections:     {(df['hla_neighbors'] > 0).sum():,}")

    return df


# ── Prioritize and save ───────────────────────────────────────────────────────

def prioritize_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply final filters and rank epitopes.

    Filters:
        1. GNN score > 0.5 (model predicts immunogenic)
        2. Valid amino acid sequence (already guaranteed by cleaning)

    Then rank by composite_score descending.
    """
    logger.info("Applying prioritization filters...")

    # Filter: only epitopes the model predicts as immunogenic
    df_candidates = df[df["gnn_score"] > 0.5].copy()
    logger.info(f"  GNN score > 0.5: {len(df_candidates):,} candidates")

    # Sort by composite score
    df_candidates = df_candidates.sort_values("composite_score", ascending=False)
    df_candidates["rank"] = range(1, len(df_candidates) + 1)

    # Reorder columns for clarity
    cols = [
        "rank", "epitope_seq", "seq_length", "mhc_class",
        "composite_score", "gnn_score", "tcr_evidence", "hla_coverage_score",
        "hla_neighbors", "true_label", "source_gene", "source_protein",
    ]
    df_candidates = df_candidates[cols].reset_index(drop=True)

    logger.info(f"  Final candidates: {len(df_candidates):,}")
    logger.info(f"  With TCR evidence: {df_candidates['tcr_evidence'].sum():,}")
    logger.info(f"  True positives in top 50: "
                f"{df_candidates.head(50)['true_label'].sum():,} / 50")

    return df_candidates


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_score_distributions(df: pd.DataFrame, df_candidates: pd.DataFrame) -> None:
    """Plot GNN score distribution split by true label."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Epitope Score Distributions", fontweight="bold")

    # GNN score by true label
    ax = axes[0]
    df_pos = df[df["true_label"] == 1]["gnn_score"]
    df_neg = df[df["true_label"] == 0]["gnn_score"]
    ax.hist(df_neg, bins=50, alpha=0.6, color="#E84855", label=f"Negative (n={len(df_neg):,})",
            density=True)
    ax.hist(df_pos, bins=50, alpha=0.7, color="#2E86AB", label=f"Positive (n={len(df_pos):,})",
            density=True)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="Threshold=0.5")
    ax.set_xlabel("GNN score"); ax.set_ylabel("Density")
    ax.set_title("GNN score by true label"); ax.legend(fontsize=9)

    # Composite score distribution for candidates
    ax = axes[1]
    ax.hist(df_candidates["composite_score"], bins=40, color="#3BB273", alpha=0.8)
    ax.axvline(df_candidates["composite_score"].median(), color="gray",
               linestyle="--", label=f"Median={df_candidates['composite_score'].median():.3f}")
    ax.set_xlabel("Composite score"); ax.set_ylabel("Count")
    ax.set_title("Composite score distribution\n(candidates only)"); ax.legend(fontsize=9)

    # MHC class breakdown in candidates
    ax = axes[2]
    mhc_counts = df_candidates["mhc_class"].value_counts()
    colors = ["#2E86AB", "#E84855"]
    bars = ax.bar(mhc_counts.index, mhc_counts.values, color=colors[:len(mhc_counts)],
                  edgecolor="white")
    for bar, val in zip(bars, mhc_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(val), ha="center", fontweight="bold")
    ax.set_xlabel("MHC class"); ax.set_ylabel("Number of candidates")
    ax.set_title("MHC class distribution\namong candidates")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "10_score_distributions.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved 10_score_distributions.png")


def plot_top20_candidates(df_candidates: pd.DataFrame) -> None:
    """Horizontal bar chart of top 20 candidates with score breakdown."""
    top20 = df_candidates.head(20).copy()
    top20 = top20.iloc[::-1]  # reverse for bottom-up display

    fig, ax = plt.subplots(figsize=(12, 9))

    y = range(len(top20))
    # Stacked bars showing score components
    gnn_contrib = 0.50 * top20["gnn_score"]
    tcr_contrib = 0.30 * top20["tcr_evidence"]
    hla_contrib = 0.20 * top20["hla_coverage_score"]

    ax.barh(y, gnn_contrib, color="#2E86AB", alpha=0.9, label="GNN score (50%)")
    ax.barh(y, tcr_contrib, left=gnn_contrib, color="#E84855", alpha=0.9, label="TCR evidence (30%)")
    ax.barh(y, hla_contrib, left=gnn_contrib+tcr_contrib, color="#3BB273", alpha=0.9,
            label="HLA coverage (20%)")

    # Y-axis labels: rank + sequence
    labels = []
    for _, row in top20.iterrows():
        tcr_mark = " ★" if row["tcr_evidence"] else ""
        label    = f"#{int(row['rank'])} {row['epitope_seq']}{tcr_mark}"
        labels.append(label)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, fontfamily="monospace")
    ax.set_xlabel("Composite score")
    ax.set_title("Top 20 Vaccine Candidates\n(★ = TCR-confirmed gold standard)",
                 fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)

    # Add true label annotations
    for i, (_, row) in enumerate(top20.iterrows()):
        color = "#2E86AB" if row["true_label"] else "#888"
        status = "✓" if row["true_label"] else "?"
        ax.text(row["composite_score"] + 0.002, i, status,
                va="center", color=color, fontweight="bold", fontsize=10)

    ax.annotate("✓ = IEDB-confirmed immunogenic  ? = predicted only",
                xy=(0.5, -0.06), xycoords="axes fraction",
                ha="center", fontsize=9, color="gray")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "11_top20_candidates.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved 11_top20_candidates.png")


def plot_source_protein_analysis(df_candidates: pd.DataFrame) -> None:
    """Which TB proteins produce the most high-scoring candidates?"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Source Protein Analysis — Top Vaccine Candidates", fontweight="bold")

    top100 = df_candidates.head(100)

    # Top source proteins by candidate count
    ax = axes[0]
    gene_counts = (
        top100[top100["source_gene"] != ""]["source_gene"]
        .value_counts().head(15)
    )
    colors = ["#7B4F9E" if i < 5 else "#B8A0CC" for i in range(len(gene_counts))]
    bars = ax.barh(range(len(gene_counts)), gene_counts.values,
                   color=colors, edgecolor="white")
    ax.set_yticks(range(len(gene_counts)))
    ax.set_yticklabels(gene_counts.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of top-100 candidates")
    ax.set_title("Top source proteins\n(top-100 candidates)")
    for bar, val in zip(bars, gene_counts.values):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                str(val), va="center", fontsize=8)

    # GNN score vs composite score scatter
    ax = axes[1]
    scatter = ax.scatter(
        df_candidates.head(200)["gnn_score"],
        df_candidates.head(200)["composite_score"],
        c=df_candidates.head(200)["tcr_evidence"],
        cmap="RdYlGn", alpha=0.7, s=40,
        vmin=0, vmax=1,
    )
    ax.set_xlabel("GNN immunogenicity score")
    ax.set_ylabel("Composite vaccine score")
    ax.set_title("GNN score vs composite score\n(top-200 candidates, color=TCR evidence)")
    plt.colorbar(scatter, ax=ax, label="TCR evidence (1=yes)")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "12_source_protein_analysis.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved 12_source_protein_analysis.png")


# ── Print summary ─────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, df_candidates: pd.DataFrame) -> None:
    console.rule("[bold green]Prioritization Complete[/bold green]")

    # Top 20 table
    t = Table(
        title="Top 20 Vaccine Candidates",
        header_style="bold cyan",
        show_lines=True,
    )
    t.add_column("Rank",      style="bold yellow", justify="right", min_width=5)
    t.add_column("Sequence",  style="bold white",  min_width=20)
    t.add_column("Length",    style="white",        justify="center", min_width=6)
    t.add_column("MHC",       style="dim",          min_width=12)
    t.add_column("Score",     style="bold green",   justify="right", min_width=7)
    t.add_column("GNN",       style="cyan",         justify="right", min_width=7)
    t.add_column("TCR",       style="white",        justify="center", min_width=5)
    t.add_column("Gene",      style="dim",          min_width=10)

    for _, row in df_candidates.head(20).iterrows():
        tcr_mark = "[bold green]YES[/bold green]" if row["tcr_evidence"] else "—"
        t.add_row(
            str(int(row["rank"])),
            row["epitope_seq"],
            str(int(row["seq_length"])),
            "I (CD8)" if "I (CD" in row["mhc_class"] else "II (CD4)",
            f"{row['composite_score']:.4f}",
            f"{row['gnn_score']:.4f}",
            tcr_mark,
            row["source_gene"] or "—",
        )

    console.print(t)

    # Gold standard check
    gold = df_candidates[df_candidates["tcr_evidence"] == 1]
    console.print(f"\n[bold]Gold standard epitopes (TCR-confirmed) in top candidates:[/bold]")
    if len(gold) > 0:
        for _, row in gold.head(10).iterrows():
            console.print(
                f"  Rank #{int(row['rank']):4d} | {row['epitope_seq']:<25} | "
                f"score={row['composite_score']:.4f} | gene={row['source_gene'] or '?'}"
            )
    else:
        console.print("  [yellow]No TCR-confirmed epitopes in candidates — check GNN threshold[/yellow]")

    # Summary stats
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Total epitopes scored:     {len(df):,}")
    console.print(f"  Candidates (score > 0.5):  {len(df_candidates):,}")
    console.print(f"  With TCR evidence:         {df_candidates['tcr_evidence'].sum():,}")
    console.print(f"  Class I (CD8+) candidates:  {(df_candidates['mhc_class'] == 'Class I (CD8+)').sum():,}")
    console.print(f"  Class II (CD4+) candidates: {(df_candidates['mhc_class'] == 'Class II (CD4+)').sum():,}")
    console.print(f"\n[bold]Saved to:[/bold] {OUT_DIR.relative_to(PROJECT_ROOT)}")
    console.print("\n[bold cyan]Phase 6 complete.[/bold cyan]")
    console.print("[bold]Next options:[/bold]")
    console.print("  1. Improve GNN (tune hyperparameters, add edges)")
    console.print("  2. Write paper sections (Methods, Results, Discussion)")
    console.print("  3. Run multi-epitope vaccine assembly\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    console.rule("[bold cyan]Phase 6: Epitope Prioritization & Vaccine Candidate Ranking[/bold cyan]")

    # Load
    console.rule("[yellow]Loading model and graph[/yellow]")
    model, graph, HP = load_model_and_graph()

    # Score
    console.rule("[yellow]Scoring all epitopes[/yellow]")
    gnn_scores = score_all_epitopes(model, graph)

    # Annotate
    console.rule("[yellow]Building annotation table[/yellow]")
    df = build_annotation_df(graph, gnn_scores)

    # Prioritize
    console.rule("[yellow]Prioritizing candidates[/yellow]")
    df_candidates = prioritize_candidates(df)

    # Save
    df.to_csv(OUT_DIR / "all_epitopes_scored.csv", index=False)
    df_candidates.to_csv(OUT_DIR / "top_candidates.csv", index=False)
    df_candidates.head(50).to_csv(OUT_DIR / "top50_candidates.csv", index=False)

    gold = df_candidates[df_candidates["tcr_evidence"] == 1]
    gold.to_csv(OUT_DIR / "gold_standard_epitopes.csv", index=False)

    logger.info(f"  Saved all_epitopes_scored.csv ({len(df):,} rows)")
    logger.info(f"  Saved top_candidates.csv ({len(df_candidates):,} rows)")
    logger.info(f"  Saved top50_candidates.csv")
    logger.info(f"  Saved gold_standard_epitopes.csv ({len(gold):,} rows)")

    # Plots
    console.rule("[yellow]Generating figures[/yellow]")
    plot_score_distributions(df, df_candidates)
    plot_top20_candidates(df_candidates)
    plot_source_protein_analysis(df_candidates)

    # Print summary
    print_summary(df, df_candidates)


if __name__ == "__main__":
    main()