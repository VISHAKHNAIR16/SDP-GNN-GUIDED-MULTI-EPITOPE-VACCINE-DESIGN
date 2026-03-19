"""
04_build_graph.py
=================
Phase 4: Build the Heterogeneous Graph
GNN-Guided Multi-Epitope Vaccine Design

What this script does:
    Takes the ESM-2 embeddings from Phase 3 and connects them into a
    single heterogeneous graph using PyTorch Geometric (PyG).

    A heterogeneous graph has MULTIPLE node types and MULTIPLE edge types.
    This is different from a standard graph where all nodes are the same.

    Our graph has 4 node types:
        epitope   — TB peptide sequences (positive + negative)
        protein   — TB proteome (21,008 proteins)
        hla       — HLA alleles (2,000 sampled)
        tcr       — TCR sequences from VDJdb (57 unique CDR3s)

    And 4 edge types (biological relationships):
        (protein,  source_of,    epitope)  — protein contains this peptide
        (epitope,  binds_to,     hla)      — epitope-HLA binding (from IEDB)
        (epitope,  recognized_by,tcr)      — TCR recognizes this epitope (VDJdb)
        (epitope,  similar_to,   epitope)  — sequence similarity edge (k-NN)

    Why a heterogeneous graph?
        In a homogeneous graph, the GNN would treat a protein node and
        an epitope node identically. But they are biologically very different
        — proteins are 200+ aa, epitopes are 9-15 aa, they have different
        roles. A heterogeneous GNN uses separate weight matrices for each
        node type and edge type, allowing it to learn type-specific patterns.

    Output:
        data/processed/graph/heterogeneous_graph.pt   — PyG HeteroData object
        data/processed/graph/graph_stats.json         — statistics for paper

Run from project root:
    uv run python scripts/04_build_graph.py
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from loguru import logger
from rich.console import Console
from rich.table import Table

# ── Setup ─────────────────────────────────────────────────────────────────────

console = Console()

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EMBED_DIR     = PROCESSED_DIR / "embeddings"
GRAPH_DIR     = PROCESSED_DIR / "graph"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
logger.add(PROJECT_ROOT / "outputs" / "graph_building.log", rotation="1 MB")

# ── Constants ─────────────────────────────────────────────────────────────────

# k-NN similarity edges: connect each epitope to its k most similar epitopes
# based on cosine similarity of ESM-2 embeddings
KNN_K = 5

# Minimum sequence identity to draw a similarity edge (cosine similarity)
SIM_THRESHOLD = 0.85

# ── Load all embeddings and metadata ─────────────────────────────────────────

def load_all_data() -> dict:
    """Load embeddings and metadata for all node types."""
    console.rule("[bold cyan]Loading embeddings and metadata[/bold cyan]")

    data = {}

    # ── Epitopes (positive + negative combined) ──
    emb_pos  = np.load(str(EMBED_DIR / "epitopes_positive.npy"))
    emb_neg  = np.load(str(EMBED_DIR / "epitopes_negative.npy"))
    meta_pos = pd.read_csv(EMBED_DIR / "epitopes_positive_meta.csv")
    meta_neg = pd.read_csv(EMBED_DIR / "epitopes_negative_meta.csv")

    # Stack positive and negative into one matrix
    emb_epitopes  = np.vstack([emb_pos, emb_neg])
    meta_epitopes = pd.concat([meta_pos, meta_neg], ignore_index=True)
    meta_epitopes["global_idx"] = range(len(meta_epitopes))

    data["epitope"] = {
        "embeddings": emb_epitopes,
        "meta":       meta_epitopes,
        "n":          len(meta_epitopes),
    }
    logger.info(f"  Epitopes: {len(meta_epitopes):,} nodes, embedding dim={emb_epitopes.shape[1]}")

    # ── TB Proteins ──
    emb_prot  = np.load(str(EMBED_DIR / "tb_proteins.npy"))
    meta_prot = pd.read_csv(EMBED_DIR / "tb_proteins_meta.csv")

    data["protein"] = {
        "embeddings": emb_prot,
        "meta":       meta_prot,
        "n":          len(meta_prot),
    }
    logger.info(f"  TB proteins: {len(meta_prot):,} nodes, embedding dim={emb_prot.shape[1]}")

    # ── HLA sequences ──
    emb_hla  = np.load(str(EMBED_DIR / "hla_sample.npy"))
    meta_hla = pd.read_csv(EMBED_DIR / "hla_sample_meta.csv")

    data["hla"] = {
        "embeddings": emb_hla,
        "meta":       meta_hla,
        "n":          len(meta_hla),
    }
    logger.info(f"  HLA alleles: {len(meta_hla):,} nodes, embedding dim={emb_hla.shape[1]}")

    # ── TCR sequences from VDJdb ──
    # TCR CDR3 sequences need to be embedded too
    # We use the CDR3 amino acid sequence directly as input
    vjdb_path = PROCESSED_DIR / "vjdb_tb_human_clean.tsv"
    df_vjdb   = pd.read_csv(vjdb_path, sep="\t")

    # Get unique CDR3 sequences
    unique_cdr3 = df_vjdb["cdr3"].dropna().unique()
    logger.info(f"  VDJdb: {len(df_vjdb)} TCR-epitope pairs, {len(unique_cdr3)} unique CDR3s")

    # For TCR nodes we use simple one-hot amino acid composition as features
    # (CDR3s are too short for ESM-2 to give meaningful embeddings)
    # This is a valid approach for short variable-length sequences
    AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")

    def cdr3_to_features(seq: str) -> np.ndarray:
        """
        Convert CDR3 to a feature vector.
        Uses amino acid composition (20 features) + length (1 feature).
        Simple but effective for short sequences.
        """
        seq = str(seq).upper()
        counts = np.array([seq.count(aa) for aa in AA_ORDER], dtype=np.float32)
        counts /= max(len(seq), 1)    # normalize to frequencies
        length = np.array([len(seq) / 30.0], dtype=np.float32)   # normalized length
        return np.concatenate([counts, length])   # 21-dim vector

    cdr3_features = np.array([cdr3_to_features(s) for s in unique_cdr3])
    # Pad to match embedding dim (320) with zeros for compatibility
    pad_width = emb_epitopes.shape[1] - cdr3_features.shape[1]
    cdr3_features_padded = np.pad(cdr3_features, ((0, 0), (0, pad_width)))

    meta_tcr = pd.DataFrame({
        "cdr3":      unique_cdr3,
        "embed_idx": range(len(unique_cdr3)),
    })

    data["tcr"] = {
        "embeddings": cdr3_features_padded,
        "meta":       meta_tcr,
        "n":          len(meta_tcr),
        "df_vjdb":    df_vjdb,
    }
    logger.info(f"  TCR CDR3s: {len(meta_tcr):,} nodes")

    return data


# ── Edge builders ─────────────────────────────────────────────────────────────

def build_protein_epitope_edges(data: dict) -> tuple[torch.Tensor, int]:
    """
    Edge type: (protein, source_of, epitope)

    How: Match epitope sequences back to their source protein using the
    'epitope_source_molecule' column in IEDB data, which names the TB protein
    each epitope came from.

    Biology: An epitope is literally a sub-sequence of a protein. If we know
    protein X contains the sequence KLGGALQAK, we draw an edge X → KLGGALQAK.
    This tells the GNN which protein a candidate epitope comes from.
    """
    logger.info("Building protein → epitope edges (source_of)")

    meta_epi  = data["epitope"]["meta"]
    meta_prot = data["protein"]["meta"]

    # Load the original IEDB file to get source molecule info
    iedb_pos = pd.read_csv(PROCESSED_DIR / "iedb_positive_clean.csv")
    iedb_neg = pd.read_csv(PROCESSED_DIR / "iedb_negative_clean.csv")
    iedb_all = pd.concat([iedb_pos, iedb_neg], ignore_index=True)

    # Find the source molecule column
    mol_col = next(
        (c for c in iedb_all.columns if "source_molecule" in c), None
    )

    if mol_col is None:
        logger.warning("  No source_molecule column — skipping protein→epitope edges")
        return torch.zeros((2, 0), dtype=torch.long), 0

    # Build lookup: epitope_seq → source molecule name
    seq_to_source = dict(zip(iedb_all["epitope_seq"], iedb_all[mol_col].fillna("")))

    # Build lookup: protein name → protein node index
    # We match on gene_name and partial protein_name
    prot_name_to_idx = {}
    for i, row in meta_prot.iterrows():
        if row["gene_name"]:
            prot_name_to_idx[row["gene_name"].upper()] = i
        if row["protein_name"]:
            # Use first word of protein name as key
            key = str(row["protein_name"]).split()[0].upper().rstrip(",")
            prot_name_to_idx[key] = i

    src_nodes, dst_nodes = [], []
    matched = 0

    for epi_idx, row in meta_epi.iterrows():
        source = str(seq_to_source.get(row["epitope_seq"], "")).upper()
        if not source:
            continue

        # Try to find matching protein
        protein_idx = None
        # Try gene name match first (most reliable)
        for word in source.split():
            word = word.rstrip(",.")
            if word in prot_name_to_idx:
                protein_idx = prot_name_to_idx[word]
                break

        if protein_idx is not None:
            src_nodes.append(protein_idx)
            dst_nodes.append(epi_idx)
            matched += 1

    logger.info(f"  Matched {matched:,} epitopes to source proteins")

    if not src_nodes:
        return torch.zeros((2, 0), dtype=torch.long), 0

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    return edge_index, len(src_nodes)


def build_epitope_hla_edges(data: dict) -> tuple[torch.Tensor, int]:
    """
    Edge type: (epitope, binds_to, hla)

    How: IEDB data contains MHC allele information for some epitopes.
    We match these allele names to HLA nodes in our graph.

    Biology: Each immunogenic epitope was experimentally confirmed to
    bind to one or more HLA alleles. This edge encodes that relationship.
    An epitope that binds many HLA alleles is more valuable for a vaccine
    because it works across more people.

    Note: Many epitopes won't have explicit HLA data in our cleaned file.
    For those, we use embedding similarity as a proxy (handled separately).
    """
    logger.info("Building epitope → HLA edges (binds_to)")

    meta_epi = data["epitope"]["meta"]
    meta_hla = data["hla"]["meta"]

    # Load IEDB with MHC allele info
    iedb_pos = pd.read_csv(PROCESSED_DIR / "iedb_positive_clean.csv")
    iedb_neg = pd.read_csv(PROCESSED_DIR / "iedb_negative_clean.csv")
    iedb_all = pd.concat([iedb_pos, iedb_neg], ignore_index=True)

    mhc_col = next((c for c in iedb_all.columns if "mhc" in c or "allele" in c), None)

    if mhc_col is None:
        logger.warning("  No MHC allele column in IEDB — building similarity-based HLA edges")
        # Fall back: connect each epitope to the most similar HLA node by embedding
        return build_hla_edges_by_similarity(data)

    # Build allele lookup: normalized allele name → HLA node index
    hla_allele_to_idx = {}
    for i, row in meta_hla.iterrows():
        allele_str = str(row["allele"]).upper()
        # Extract allele designation like "A*02:01"
        import re
        m = re.search(r"([A-Z0-9]+\*\d+:\d+)", allele_str)
        if m:
            hla_allele_to_idx[m.group(1)] = i

    # Build epitope seq → allele mapping
    seq_to_allele = {}
    for _, row in iedb_all.iterrows():
        allele = str(row.get(mhc_col, "")).upper().strip()
        if allele and allele != "NAN":
            seq_to_allele.setdefault(row["epitope_seq"], set()).add(allele)

    src_nodes, dst_nodes = [], []

    for epi_idx, row in meta_epi.iterrows():
        alleles = seq_to_allele.get(row["epitope_seq"], set())
        for allele in alleles:
            import re
            m = re.search(r"([A-Z0-9]+\*\d+:\d+)", allele)
            if m:
                key = m.group(1)
                if key in hla_allele_to_idx:
                    src_nodes.append(epi_idx)
                    dst_nodes.append(hla_allele_to_idx[key])

    logger.info(f"  Built {len(src_nodes):,} epitope→HLA edges from allele names")

    # If very few matches, supplement with similarity-based edges
    if len(src_nodes) < 100:
        logger.info("  Few allele matches — supplementing with similarity-based edges")
        sim_src, sim_dst, _ = build_hla_edges_by_similarity(data, return_raw=True)
        src_nodes.extend(sim_src)
        dst_nodes.extend(sim_dst)
        logger.info(f"  Total after supplement: {len(src_nodes):,} edges")

    if not src_nodes:
        return torch.zeros((2, 0), dtype=torch.long), 0

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    return edge_index, len(src_nodes)


def build_hla_edges_by_similarity(data: dict, return_raw: bool = False):
    """
    Fallback: build epitope→HLA edges based on embedding similarity.
    Each epitope connects to its top-3 most similar HLA alleles.
    """
    emb_epi = data["epitope"]["embeddings"].astype(np.float32)
    emb_hla = data["hla"]["embeddings"].astype(np.float32)

    # Normalize for cosine similarity
    emb_epi_norm = emb_epi / (np.linalg.norm(emb_epi, axis=1, keepdims=True) + 1e-8)
    emb_hla_norm = emb_hla / (np.linalg.norm(emb_hla, axis=1, keepdims=True) + 1e-8)

    src_nodes, dst_nodes = [], []
    batch_size = 500

    for i in range(0, len(emb_epi_norm), batch_size):
        batch = emb_epi_norm[i:i+batch_size]
        sims  = batch @ emb_hla_norm.T   # (batch, n_hla)
        top3  = np.argsort(sims, axis=1)[:, -3:]

        for local_idx, hla_indices in enumerate(top3):
            epi_idx = i + local_idx
            for hla_idx in hla_indices:
                if sims[local_idx, hla_idx] > 0.5:   # minimum similarity
                    src_nodes.append(epi_idx)
                    dst_nodes.append(int(hla_idx))

    if return_raw:
        return src_nodes, dst_nodes, len(src_nodes)

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    return edge_index, len(src_nodes)


def build_epitope_tcr_edges(data: dict) -> tuple[torch.Tensor, int]:
    """
    Edge type: (epitope, recognized_by, tcr)

    How: Direct lookup from VDJdb — each row says "CDR3 sequence X
    recognizes epitope Y". We match both to their node indices.

    Biology: This is the most direct evidence of immune recognition.
    If a TCR sequence is known to bind an epitope, the epitope is
    almost certainly presented by HLA and immunogenic. These edges
    are the most trusted in the entire graph.
    """
    logger.info("Building epitope → TCR edges (recognized_by)")

    meta_epi = data["epitope"]["meta"]
    meta_tcr = data["tcr"]["meta"]
    df_vjdb  = data["tcr"]["df_vjdb"]

    # Build lookups
    epi_seq_to_idx = dict(zip(
        meta_epi["epitope_seq"].str.upper(),
        meta_epi["global_idx"]
    ))
    cdr3_to_idx = dict(zip(
        meta_tcr["cdr3"].str.upper(),
        meta_tcr["embed_idx"]
    ))

    src_nodes, dst_nodes = [], []

    for _, row in df_vjdb.iterrows():
        epi_seq = str(row.get("epitope", "")).upper().strip()
        cdr3    = str(row.get("cdr3", "")).upper().strip()

        epi_idx = epi_seq_to_idx.get(epi_seq)
        tcr_idx = cdr3_to_idx.get(cdr3)

        if epi_idx is not None and tcr_idx is not None:
            src_nodes.append(epi_idx)
            dst_nodes.append(tcr_idx)

    logger.info(f"  Built {len(src_nodes):,} epitope→TCR edges (gold standard)")

    if not src_nodes:
        return torch.zeros((2, 0), dtype=torch.long), 0

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    return edge_index, len(src_nodes)


def build_epitope_similarity_edges(data: dict) -> tuple[torch.Tensor, int]:
    """
    Edge type: (epitope, similar_to, epitope)

    How: Compute pairwise cosine similarity between ALL epitope embeddings
    and connect each epitope to its k nearest neighbors.

    Why: This is crucial for the GNN to generalize. If epitope A is similar
    to epitope B (which is known immunogenic), the GNN can infer A may also
    be immunogenic. This is "guilt by association" — a powerful inductive
    bias for biological sequences.

    Implementation: We use approximate k-NN for speed (exact k-NN on
    23,884 × 23,884 = 570M pairs would be too slow).
    We process in batches to avoid memory issues.
    """
    logger.info(f"Building epitope similarity edges (k={KNN_K}, threshold={SIM_THRESHOLD})")

    emb = data["epitope"]["embeddings"].astype(np.float32)
    n   = len(emb)

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    emb_norm = emb / norms

    src_nodes, dst_nodes = [], []
    batch_size = 1000   # process 1000 epitopes at a time

    logger.info(f"  Processing {n:,} epitopes in batches of {batch_size}...")

    for i in range(0, n, batch_size):
        batch     = emb_norm[i:i+batch_size]            # (batch, dim)
        sims      = batch @ emb_norm.T                   # (batch, n) cosine similarities
        sims_copy = sims.copy()
        # Zero out self-similarity and already-processed pairs
        np.fill_diagonal(sims_copy[0:len(batch), i:i+len(batch)], 0)

        # Get top-k indices for each epitope in batch
        top_k_indices = np.argsort(sims_copy, axis=1)[:, -KNN_K:]

        for local_idx in range(len(batch)):
            global_idx = i + local_idx
            for neighbor_idx in top_k_indices[local_idx]:
                sim_val = float(sims_copy[local_idx, neighbor_idx])
                if sim_val >= SIM_THRESHOLD and neighbor_idx != global_idx:
                    src_nodes.append(global_idx)
                    dst_nodes.append(int(neighbor_idx))

        if (i // batch_size) % 5 == 0:
            logger.info(f"  Progress: {min(i+batch_size, n):,}/{n:,} "
                        f"({100*min(i+batch_size,n)/n:.0f}%) — "
                        f"{len(src_nodes):,} edges so far")

    logger.info(f"  Built {len(src_nodes):,} epitope similarity edges")

    if not src_nodes:
        return torch.zeros((2, 0), dtype=torch.long), 0

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    return edge_index, len(src_nodes)


# ── Assemble HeteroData object ────────────────────────────────────────────────

def build_hetero_graph(data: dict) -> HeteroData:
    """
    Assemble all nodes and edges into a PyG HeteroData object.

    HeteroData is PyG's container for heterogeneous graphs.
    It stores node features and edge indices per type.

    Structure:
        graph['epitope'].x         — (23884, 320) float tensor
        graph['epitope'].y         — (23884,)     int tensor (0/1 labels)
        graph['protein'].x         — (21008, 320) float tensor
        graph['hla'].x             — (2000,  320) float tensor
        graph['tcr'].x             — (57,    320) float tensor
        graph['protein','source_of','epitope'].edge_index    — (2, E1)
        graph['epitope','binds_to','hla'].edge_index         — (2, E2)
        graph['epitope','recognized_by','tcr'].edge_index    — (2, E3)
        graph['epitope','similar_to','epitope'].edge_index   — (2, E4)
    """
    console.rule("[bold cyan]Assembling heterogeneous graph[/bold cyan]")

    graph = HeteroData()

    # ── Node features ──
    logger.info("Adding node features...")

    graph["epitope"].x = torch.tensor(
        data["epitope"]["embeddings"], dtype=torch.float32
    )
    graph["epitope"].y = torch.tensor(
        data["epitope"]["meta"]["label"].values, dtype=torch.long
    )
    # Store sequence strings for later analysis
    graph["epitope"].seq = data["epitope"]["meta"]["epitope_seq"].tolist()

    graph["protein"].x = torch.tensor(
        data["protein"]["embeddings"], dtype=torch.float32
    )
    graph["protein"].gene_name = data["protein"]["meta"]["gene_name"].tolist()

    graph["hla"].x = torch.tensor(
        data["hla"]["embeddings"], dtype=torch.float32
    )
    graph["hla"].allele = data["hla"]["meta"]["allele"].tolist()

    graph["tcr"].x = torch.tensor(
        data["tcr"]["embeddings"], dtype=torch.float32
    )
    graph["tcr"].cdr3 = data["tcr"]["meta"]["cdr3"].tolist()

    logger.info(f"  epitope nodes: {graph['epitope'].x.shape}")
    logger.info(f"  protein nodes: {graph['protein'].x.shape}")
    logger.info(f"  hla nodes:     {graph['hla'].x.shape}")
    logger.info(f"  tcr nodes:     {graph['tcr'].x.shape}")

    # ── Edges ──
    logger.info("Building edges...")

    ei_prot_epi, n_prot_epi = build_protein_epitope_edges(data)
    graph["protein", "source_of", "epitope"].edge_index = ei_prot_epi

    ei_epi_hla, n_epi_hla = build_epitope_hla_edges(data)
    graph["epitope", "binds_to", "hla"].edge_index = ei_epi_hla

    ei_epi_tcr, n_epi_tcr = build_epitope_tcr_edges(data)
    graph["epitope", "recognized_by", "tcr"].edge_index = ei_epi_tcr

    ei_epi_sim, n_epi_sim = build_epitope_similarity_edges(data)
    graph["epitope", "similar_to", "epitope"].edge_index = ei_epi_sim

    return graph, {
        "n_epitopes":    data["epitope"]["n"],
        "n_proteins":    data["protein"]["n"],
        "n_hla":         data["hla"]["n"],
        "n_tcr":         data["tcr"]["n"],
        "e_source_of":   n_prot_epi,
        "e_binds_to":    n_epi_hla,
        "e_recognized":  n_epi_tcr,
        "e_similar_to":  n_epi_sim,
        "n_positive":    int(data["epitope"]["meta"]["label"].sum()),
        "n_negative":    int((data["epitope"]["meta"]["label"] == 0).sum()),
    }


# ── Save and validate ─────────────────────────────────────────────────────────

def save_graph(graph: HeteroData, stats: dict) -> None:
    """Save the graph object and statistics."""
    graph_path = GRAPH_DIR / "heterogeneous_graph.pt"
    torch.save(graph, str(graph_path))
    size_mb = graph_path.stat().st_size / 1e6
    logger.info(f"  Graph saved: {graph_path.relative_to(PROJECT_ROOT)} ({size_mb:.1f} MB)")

    stats_path = GRAPH_DIR / "graph_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"  Stats saved: {stats_path.relative_to(PROJECT_ROOT)}")


def validate_graph(graph: HeteroData, stats: dict) -> None:
    """Print a summary table and run basic validation checks."""
    console.rule("[bold green]Graph Validation[/bold green]")

    t = Table(title="Heterogeneous Graph Summary", header_style="bold cyan", show_lines=True)
    t.add_column("Component",    style="white",       min_width=35)
    t.add_column("Count",        style="bold yellow",  min_width=12)
    t.add_column("Details",      style="dim",           min_width=30)

    t.add_row("Epitope nodes",   f"{stats['n_epitopes']:,}",
              f"{stats['n_positive']:,} pos / {stats['n_negative']:,} neg")
    t.add_row("Protein nodes",   f"{stats['n_proteins']:,}", "TB H37Rv proteome")
    t.add_row("HLA nodes",       f"{stats['n_hla']:,}",      "Stratified sample")
    t.add_row("TCR nodes",       f"{stats['n_tcr']:,}",      "Unique CDR3 sequences")
    t.add_row("─" * 30, "─" * 10, "─" * 25)

    total_nodes = stats['n_epitopes'] + stats['n_proteins'] + stats['n_hla'] + stats['n_tcr']
    t.add_row("Total nodes", f"{total_nodes:,}", "Across all types")

    t.add_row("protein→epitope edges",  f"{stats['e_source_of']:,}",  "source_of relation")
    t.add_row("epitope→HLA edges",      f"{stats['e_binds_to']:,}",   "binds_to relation")
    t.add_row("epitope→TCR edges",      f"{stats['e_recognized']:,}", "recognized_by (gold)")
    t.add_row("epitope→epitope edges",  f"{stats['e_similar_to']:,}", "similarity k-NN")
    t.add_row("─" * 30, "─" * 10, "─" * 25)

    total_edges = (stats['e_source_of'] + stats['e_binds_to'] +
                   stats['e_recognized'] + stats['e_similar_to'])
    t.add_row("Total edges", f"{total_edges:,}", "Across all types")

    console.print(t)

    # Checks
    issues = []
    if stats["e_source_of"] == 0:
        issues.append("No protein→epitope edges — source molecule matching failed")
    if stats["e_recognized"] == 0:
        issues.append("No epitope→TCR edges — VDJdb matching failed")
    if stats["e_similar_to"] < 1000:
        issues.append(f"Very few similarity edges ({stats['e_similar_to']}) — check threshold")

    console.print()
    if issues:
        console.print("[bold red]Warnings:[/bold red]")
        for issue in issues:
            console.print(f"  [yellow]• {issue}[/yellow]")
    else:
        console.print("[bold green]Graph looks healthy![/bold green]")

    console.print(
        "\n[bold cyan]Next step:[/bold cyan] "
        "uv run python scripts/05_train_gnn.py\n"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    console.rule("[bold cyan]Phase 4: Building Heterogeneous Graph[/bold cyan]")
    console.print(
        "\n[bold]Framework:[/bold] PyTorch Geometric (PyG)\n"
        "[bold]Node types:[/bold] epitope, protein, hla, tcr\n"
        "[bold]Edge types:[/bold] source_of, binds_to, recognized_by, similar_to\n"
    )

    t0   = time.time()
    data = load_all_data()

    graph, stats = build_hetero_graph(data)
    save_graph(graph, stats)
    validate_graph(graph, stats)

    logger.info(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()