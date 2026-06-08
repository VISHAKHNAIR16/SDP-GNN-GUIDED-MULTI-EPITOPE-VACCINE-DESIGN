"""
04b_fix_graph_covid.py
======================
Pre-training fix: clean two issues in the COVID graph before script 05.

Issue 1 — Orphan protein nodes (6 of 17 proteins have zero epitope edges)
    Cause: gene name case variants (orf1ab vs ORF3a vs 3a) created duplicate
    protein nodes. The edge builder matched epitopes to the lowercase/alternate
    variant, leaving the canonical-named duplicate as an orphan.

    Fix: rebuild covid_proteins_reference.csv with proper deduplication,
    re-embed only the changed proteins, rebuild the graph.

    Faster fix applied here: remove orphan nodes from the saved graph and
    reindex all protein-related edges. Saves re-running scripts 1b and 3.

Issue 2 — tcr_confirmed flag set on 1,017 nodes instead of 668
    Cause: gold-standard computation matched epitope sequences across both
    positive and negative labels. 349 negative epitopes share sequences
    with VDJdb entries but are labelled non-immunogenic in IEDB.

    Fix: restrict tcr_confirmed=1 to positive epitopes (label=1) only.
    A negative epitope cannot be gold-standard by definition — if IEDB
    called it non-immunogenic, it doesn't get the dual-evidence flag
    regardless of VDJdb appearance.

Output:
    Overwrites data/processed_covid/graph/covid_graph.pt with clean version
    Updates data/processed_covid/graph/covid_graph_stats.json

Run from project root:
    uv run python scripts/04b_fix_graph_covid.py
"""

import sys
import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table

# ── Setup ─────────────────────────────────────────────────────────────────────

console = Console()

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_covid"
EMBED_DIR     = PROCESSED_DIR / "embeddings"
GRAPH_DIR     = PROCESSED_DIR / "graph"

logger.remove()
logger.add(sys.stderr,
           format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")


# ── Fix 1: Remove orphan protein nodes ───────────────────────────────────────

def fix_orphan_proteins(graph, meta_prot: pd.DataFrame) -> tuple:
    """
    Remove protein nodes that have zero epitope edges, reindex the rest.

    Returns updated graph and a mapping old_idx → new_idx for proteins.
    """
    console.rule("[bold cyan]Fix 1: Removing orphan protein nodes[/bold cyan]")

    edge_index = graph["protein", "source_of", "epitope"].edge_index
    if edge_index.shape[1] == 0:
        logger.warning("  No protein→epitope edges exist — nothing to fix")
        return graph, {}, meta_prot

    # Find which protein indices actually appear in edges
    active_protein_idxs = torch.unique(edge_index[0]).tolist()
    all_protein_idxs    = list(range(graph["protein"].x.shape[0]))
    orphan_idxs         = [i for i in all_protein_idxs if i not in active_protein_idxs]

    logger.info(f"  Total protein nodes    : {len(all_protein_idxs)}")
    logger.info(f"  Active (have edges)    : {len(active_protein_idxs)}")
    logger.info(f"  Orphan (no edges)      : {len(orphan_idxs)}")

    if not orphan_idxs:
        logger.info("  No orphans found — nothing to remove")
        return graph, {i: i for i in all_protein_idxs}, meta_prot

    # Log what we're removing
    for idx in orphan_idxs:
        gene = meta_prot.iloc[idx]["gene_name"]
        name = meta_prot.iloc[idx]["protein_name"][:40]
        logger.info(f"  Removing orphan: index {idx} ({gene} | {name})")

    # Build old → new index mapping (only for active proteins)
    old_to_new = {}
    new_idx = 0
    for old_idx in sorted(active_protein_idxs):
        old_to_new[old_idx] = new_idx
        new_idx += 1

    # Update protein node features (keep only active rows)
    active_tensor_idxs = torch.tensor(sorted(active_protein_idxs), dtype=torch.long)
    graph["protein"].x = graph["protein"].x[active_tensor_idxs]

    # Update gene_name attribute
    if hasattr(graph["protein"], "gene_name"):
        graph["protein"].gene_name = [
            graph["protein"].gene_name[i] for i in sorted(active_protein_idxs)
        ]

    # Reindex protein→epitope edge_index source nodes
    old_src = edge_index[0]
    new_src = torch.tensor(
        [old_to_new[int(i)] for i in old_src.tolist()],
        dtype=torch.long
    )
    graph["protein", "source_of", "epitope"].edge_index = torch.stack(
        [new_src, edge_index[1]]
    )

    # Update metadata
    meta_prot_clean = meta_prot.iloc[sorted(active_protein_idxs)].reset_index(drop=True)

    logger.info(
        f"  Protein nodes after fix: {graph['protein'].x.shape[0]} "
        f"(removed {len(orphan_idxs)} orphans)"
    )

    return graph, old_to_new, meta_prot_clean


# ── Fix 2: Correct tcr_confirmed flag ────────────────────────────────────────

def fix_tcr_confirmed(graph) -> int:
    """
    Restrict tcr_confirmed=1 to positive epitopes only (label=1).

    A negative epitope (label=0) cannot be gold-standard regardless of
    whether its sequence appears in VDJdb. If IEDB called it non-immunogenic,
    the dual-evidence flag is meaningless and misleading for the GNN.

    Returns the corrected count of gold-standard nodes.
    """
    console.rule("[bold cyan]Fix 2: Correcting tcr_confirmed flag[/bold cyan]")

    labels        = graph["epitope"].y              # (N,) — 0 or 1
    tcr_confirmed = graph["epitope"].tcr_confirmed  # (N,) — 0 or 1

    before = int(tcr_confirmed.sum())

    # Zero out the flag for any negative epitope
    corrected = tcr_confirmed.clone()
    corrected[labels == 0] = 0

    after = int(corrected.sum())

    logger.info(f"  tcr_confirmed before fix : {before:,}")
    logger.info(f"  Negatives with flag=1    : {before - after:,} (incorrectly set)")
    logger.info(f"  tcr_confirmed after fix  : {after:,}")
    logger.info(
        f"  These {after:,} are the true gold-standard epitopes "
        f"(positive in IEDB + confirmed in VDJdb)"
    )

    graph["epitope"].tcr_confirmed = corrected
    return after


# ── Validate fixed graph ──────────────────────────────────────────────────────

def validate_fixed_graph(graph, meta_prot_clean: pd.DataFrame,
                         n_gold: int) -> dict:
    console.rule("[bold green]Fixed Graph Validation[/bold green]")

    t = Table(title="COVID Graph — After Fixes",
              header_style="bold cyan", show_lines=True)
    t.add_column("Component",  style="white",      min_width=40)
    t.add_column("Count",      style="bold yellow", min_width=12)
    t.add_column("Details",    style="dim",          min_width=30)

    n_epi  = graph["epitope"].x.shape[0]
    n_prot = graph["protein"].x.shape[0]
    n_hla  = graph["hla"].x.shape[0]
    n_tcr  = graph["tcr"].x.shape[0]
    n_pos  = int((graph["epitope"].y == 1).sum())
    n_neg  = int((graph["epitope"].y == 0).sum())

    e_src  = graph["protein", "source_of",    "epitope"].edge_index.shape[1]
    e_hla  = graph["epitope", "binds_to",     "hla"].edge_index.shape[1]
    e_tcr  = graph["epitope", "recognized_by","tcr"].edge_index.shape[1]
    e_sim  = graph["epitope", "similar_to",   "epitope"].edge_index.shape[1]

    t.add_row("Epitope nodes", f"{n_epi:,}", f"{n_pos:,} pos / {n_neg:,} neg")
    t.add_row("  gold-standard (tcr_confirmed=1)",
              f"{n_gold:,}", "positive only — fixed")
    t.add_row("Protein nodes", f"{n_prot:,}",
              "orphans removed")
    t.add_row("HLA nodes",     f"{n_hla:,}", "unchanged")
    t.add_row("TCR nodes",     f"{n_tcr:,}", "unchanged")
    t.add_row("─" * 35, "─" * 10, "─" * 25)
    t.add_row("Total nodes",
              f"{n_epi + n_prot + n_hla + n_tcr:,}", "")
    t.add_row("protein → epitope", f"{e_src:,}", "source_of")
    t.add_row("epitope → HLA",     f"{e_hla:,}", "binds_to")
    t.add_row("epitope → TCR",     f"{e_tcr:,}", "recognized_by")
    t.add_row("epitope → epitope", f"{e_sim:,}", "similar_to (k-NN)")
    t.add_row("─" * 35, "─" * 10, "─" * 25)
    t.add_row("Total edges",
              f"{e_src + e_hla + e_tcr + e_sim:,}", "")

    console.print(t)

    # Show active protein distribution
    console.print("\n[bold]Active protein nodes after deduplication:[/bold]")
    src_ei = graph["protein", "source_of", "epitope"].edge_index
    vals, counts = torch.unique(src_ei[0], return_counts=True)
    for v, c in zip(vals.tolist(), counts.tolist()):
        if v < len(meta_prot_clean):
            gene = meta_prot_clean.iloc[v]["gene_name"]
            name = str(meta_prot_clean.iloc[v]["protein_name"])[:30]
            console.print(
                f"  [{v:2d}] {gene:8s} | {name:30s}: {c:,} epitope edges"
            )

    stats = {
        "disease":         "COVID-19 (SARS-CoV-2)",
        "n_epitopes":      n_epi,
        "n_positive":      n_pos,
        "n_negative":      n_neg,
        "n_proteins":      n_prot,
        "n_hla":           n_hla,
        "n_tcr":           n_tcr,
        "n_gold_standard": n_gold,
        "e_source_of":     e_src,
        "e_binds_to":      e_hla,
        "e_recognized_by": e_tcr,
        "e_similar_to":    e_sim,
        "fixes_applied":   ["orphan_protein_removal", "tcr_confirmed_label_restriction"],
    }
    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    console.rule("[bold cyan]COVID Graph Fix: Pre-training cleanup[/bold cyan]")

    # Load graph
    graph_path = GRAPH_DIR / "covid_graph.pt"
    if not graph_path.exists():
        logger.error(f"Graph not found: {graph_path}")
        logger.error("Run 04_build_graph_covid.py first.")
        sys.exit(1)

    logger.info(f"  Loading graph from {graph_path.relative_to(PROJECT_ROOT)}")
    graph = torch.load(str(graph_path), weights_only=False)

    meta_prot = pd.read_csv(EMBED_DIR / "covid_proteins_meta.csv")

    logger.info(
        f"  Loaded graph: "
        f"{graph['epitope'].x.shape[0]:,} epitopes, "
        f"{graph['protein'].x.shape[0]:,} proteins"
    )

    # Apply Fix 1: remove orphan protein nodes
    graph, old_to_new, meta_prot_clean = fix_orphan_proteins(graph, meta_prot)

    # Apply Fix 2: correct tcr_confirmed flag
    n_gold = fix_tcr_confirmed(graph)

    # Validate and collect stats
    stats = validate_fixed_graph(graph, meta_prot_clean, n_gold)

    # Save — overwrite the original graph file
    torch.save(graph, str(graph_path))
    size_mb = graph_path.stat().st_size / 1e6
    logger.info(
        f"  Saved fixed graph: {graph_path.relative_to(PROJECT_ROOT)} "
        f"({size_mb:.1f} MB)"
    )

    stats_path = GRAPH_DIR / "covid_graph_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"  Stats updated: {stats_path.relative_to(PROJECT_ROOT)}")

    # Save clean protein metadata for downstream use
    clean_meta_path = EMBED_DIR / "covid_proteins_active_meta.csv"
    meta_prot_clean.to_csv(clean_meta_path, index=False)
    logger.info(
        f"  Active protein metadata: {clean_meta_path.relative_to(PROJECT_ROOT)}"
    )

    console.print(
        "\n[bold cyan]Next step:[/bold cyan] "
        "uv run python scripts/05_train_gnn_covid.py\n"
    )


if __name__ == "__main__":
    main()