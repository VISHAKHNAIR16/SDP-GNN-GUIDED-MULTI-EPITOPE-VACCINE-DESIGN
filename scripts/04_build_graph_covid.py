"""
04_build_graph_covid.py
=======================
Phase 4 (COVID Validation): Build the Heterogeneous Graph
GNN-Guided Multi-Epitope Vaccine Design

What this script does:
    Takes the COVID ESM-2 embeddings from Phase 3 and connects them into
    a heterogeneous graph using PyTorch Geometric (PyG).

    Structurally identical to the TB graph with one deliberate modification:

    TCR node strategy (Option A modified):
        TB graph: all 57 unique CDR3s as nodes (trivially small)
        COVID naive: all 9,341 unique CDR3s — too large for laptop GPU
        COVID here: only CDR3s that link to the 668 gold-standard epitopes
                    (IEDB positive ∩ VDJdb confirmed)

        Why: CDR3s that bind non-gold-standard epitopes are either:
          (a) linking to epitopes not in our IEDB positive set, OR
          (b) orphan nodes with no epitope connection in our graph
        Both cases contribute noise and memory, not signal.
        Filtering to gold-standard-linked CDR3s keeps structural parity
        with TB while keeping graph size manageable on a 6.4GB GPU.

    Graph structure:
        Node types:
            epitope  — 8,348 COVID peptides (4,213 pos + 4,135 neg)
            protein  — 17 SARS-CoV-2 reference proteins
            hla      — 16 HLA alleles (from VDJdb mhc_a)
            tcr      — CDR3s linked to gold-standard epitopes only

        Edge types:
            (protein,  source_of,    epitope)  — protein contains peptide
            (epitope,  binds_to,     hla)      — HLA allele matching
            (epitope,  recognized_by,tcr)      — VDJdb gold-standard edges
            (epitope,  similar_to,   epitope)  — k-NN cosine similarity

    Output:
        data/processed_covid/graph/covid_graph.pt
        data/processed_covid/graph/covid_graph_stats.json

Run from project root:
    uv run python scripts/04_build_graph_covid.py
"""

import sys
import json
import re
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
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_covid"
EMBED_DIR     = PROCESSED_DIR / "embeddings"
GRAPH_DIR     = PROCESSED_DIR / "graph"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stderr,
           format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
logger.add(PROJECT_ROOT / "outputs" / "graph_building_covid.log", rotation="1 MB")

# ── Constants — identical to TB pipeline for comparison validity ───────────────

KNN_K         = 5       # neighbours per epitope in similarity graph
SIM_THRESHOLD = 0.85    # minimum cosine similarity for a k-NN edge


# ── Load all embeddings and metadata ─────────────────────────────────────────

def load_all_data() -> dict:
    """
    Load COVID embeddings and metadata for all four node types.

    Key difference from TB: TCR nodes are pre-filtered to only those
    CDR3 sequences that recognise gold-standard epitopes. This is computed
    here during loading, not as a separate preprocessing step.
    """
    console.rule("[bold cyan]Loading COVID embeddings and metadata[/bold cyan]")
    data = {}

    # ── Epitopes (positive + negative combined) ──────────────────────────────
    emb_pos  = np.load(str(EMBED_DIR / "epitopes_positive_covid.npy"))
    emb_neg  = np.load(str(EMBED_DIR / "epitopes_negative_covid.npy"))
    meta_pos = pd.read_csv(EMBED_DIR / "epitopes_positive_covid_meta.csv")
    meta_neg = pd.read_csv(EMBED_DIR / "epitopes_negative_covid_meta.csv")

    emb_epitopes  = np.vstack([emb_pos, emb_neg])
    meta_epitopes = pd.concat([meta_pos, meta_neg], ignore_index=True)
    meta_epitopes["global_idx"] = range(len(meta_epitopes))

    data["epitope"] = {
        "embeddings": emb_epitopes,
        "meta":       meta_epitopes,
        "n":          len(meta_epitopes),
    }
    logger.info(
        f"  Epitopes: {len(meta_epitopes):,} nodes, dim={emb_epitopes.shape[1]}"
    )

    # ── COVID proteins ────────────────────────────────────────────────────────
    emb_prot  = np.load(str(EMBED_DIR / "covid_proteins.npy"))
    meta_prot = pd.read_csv(EMBED_DIR / "covid_proteins_meta.csv")

    data["protein"] = {
        "embeddings": emb_prot,
        "meta":       meta_prot,
        "n":          len(meta_prot),
    }
    logger.info(
        f"  COVID proteins: {len(meta_prot):,} nodes, dim={emb_prot.shape[1]}"
    )

    # ── HLA alleles ───────────────────────────────────────────────────────────
    emb_hla  = np.load(str(EMBED_DIR / "hla_covid.npy"))
    meta_hla = pd.read_csv(EMBED_DIR / "hla_covid_meta.csv")

    data["hla"] = {
        "embeddings": emb_hla,
        "meta":       meta_hla,
        "n":          len(meta_hla),
    }
    logger.info(
        f"  HLA alleles: {len(meta_hla):,} nodes, dim={emb_hla.shape[1]}"
    )

    # ── TCR CDR3s (gold-standard filtered) ───────────────────────────────────
    #
    # Strategy: load full VDJdb, compute gold-standard epitopes (IEDB pos ∩ VDJdb),
    # then keep only CDR3s that link to those gold-standard epitopes.
    #
    # This is the Option A modification: structural parity with TB (real CDR3 nodes),
    # but limited to CDR3s that have edges in the graph — no orphan nodes.

    df_vjdb = pd.read_csv(PROCESSED_DIR / "vdjdb_covid_clean.tsv", sep="\t")

    # Compute gold-standard set: IEDB positive ∩ VDJdb
    iedb_pos_seqs = set(
        meta_epitopes[meta_epitopes["label"] == 1]["epitope_seq"].str.upper()
    )
    vjdb_epi_seqs = set(df_vjdb["epitope"].str.upper().dropna())
    gold_standard  = iedb_pos_seqs & vjdb_epi_seqs

    logger.info(f"  Gold-standard epitopes (IEDB pos ∩ VDJdb): {len(gold_standard):,}")

    # Filter VDJdb to only rows where the epitope is in gold-standard set
    df_vjdb_gold = df_vjdb[
        df_vjdb["epitope"].str.upper().isin(gold_standard)
    ].copy()

    logger.info(
        f"  VDJdb rows linking to gold-standard epitopes: {len(df_vjdb_gold):,}"
    )

    # Get unique CDR3s from this filtered set
    unique_cdr3_gold = df_vjdb_gold["cdr3"].dropna().unique()
    logger.info(
        f"  Unique CDR3s linked to gold-standard epitopes: {len(unique_cdr3_gold):,} "
        f"(filtered from {df_vjdb['cdr3'].nunique():,} total)"
    )

    # Feature encoding: amino acid composition + length (21-dim), padded to 320
    # Same approach as TB — CDR3s are too short (12–20aa) for meaningful ESM-2 pooling
    AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")

    def cdr3_to_features(seq: str) -> np.ndarray:
        """
        21-dim feature vector for a CDR3 sequence.
        20 AA frequencies (normalized) + 1 length feature (normalized to max 30aa).
        Padded to 320 dims with zeros to match embedding space of other node types.
        """
        seq    = str(seq).upper()
        counts = np.array([seq.count(aa) for aa in AA_ORDER], dtype=np.float32)
        counts /= max(len(seq), 1)
        length  = np.array([len(seq) / 30.0], dtype=np.float32)
        return np.concatenate([counts, length])   # 21-dim

    cdr3_features_21 = np.array(
        [cdr3_to_features(s) for s in unique_cdr3_gold]
    )
    # Pad 21-dim → 320-dim to match epitope/protein/HLA embedding space
    pad_width         = emb_epitopes.shape[1] - cdr3_features_21.shape[1]
    cdr3_features_pad = np.pad(cdr3_features_21, ((0, 0), (0, pad_width)))

    meta_tcr = pd.DataFrame({
        "cdr3":      unique_cdr3_gold,
        "embed_idx": range(len(unique_cdr3_gold)),
    })

    data["tcr"] = {
        "embeddings":    cdr3_features_pad,
        "meta":          meta_tcr,
        "n":             len(meta_tcr),
        "df_vjdb_gold":  df_vjdb_gold,    # filtered VDJdb — used in edge builder
        "gold_standard": gold_standard,
    }
    logger.info(f"  TCR CDR3 nodes (gold-standard only): {len(meta_tcr):,}")

    return data


# ── Edge builder 1: protein → epitope ─────────────────────────────────────────

def build_protein_epitope_edges(data: dict) -> tuple[torch.Tensor, int]:
    """
    Edge type: (protein, source_of, epitope)

    Match each epitope back to its source COVID protein using the
    source_molecule column in our cleaned IEDB CSVs.

    COVID difference from TB:
        TB has 4,000+ proteins so matching is broad.
        COVID has 17 proteins so we match aggressively — any partial
        match on gene name (S, N, M, E, ORF1ab etc.) is accepted.

    Biology: YYVGYLQPRTFLL is a Spike-derived epitope. Knowing it comes
    from Spike tells the GNN that Spike-protein-neighbourhood epitopes
    share structural context. The protein→epitope edge propagates
    protein-level information down to the epitope during message passing.
    """
    logger.info("Building protein → epitope edges (source_of)")

    meta_epi  = data["epitope"]["meta"]
    meta_prot = data["protein"]["meta"]

    # Load IEDB CSVs — they have the source_molecule column
    iedb_pos = pd.read_csv(PROCESSED_DIR / "iedb_positive_covid.csv")
    iedb_neg = pd.read_csv(PROCESSED_DIR / "iedb_negative_covid.csv")
    iedb_all = pd.concat([iedb_pos, iedb_neg], ignore_index=True)

    # Build: epitope_seq → source molecule string
    seq_to_source = dict(
        zip(
            iedb_all["epitope_seq"].str.upper(),
            iedb_all["source_molecule"].fillna("").astype(str)
        )
    )

    # Build: normalised gene/protein token → protein node index
    # COVID genes are short and well-known: S, N, M, E, ORF1ab, ORF3a, etc.
    prot_token_to_idx = {}

    for idx, row in meta_prot.iterrows():
        gene = str(row.get("gene_name", "")).strip().upper()
        name = str(row.get("protein_name", "")).strip().upper()

        if gene:
            prot_token_to_idx[gene] = idx

        # Also index by first meaningful word of protein name
        for word in name.split():
            word = word.rstrip(".,;:")
            if len(word) >= 2:
                prot_token_to_idx[word] = idx

    # COVID-specific mapping: IEDB often uses verbose protein names
    # Map common IEDB terms to canonical gene names in our reference proteome
    IEDB_TO_GENE = {
        "SPIKE":          "S",
        "SURFACE":        "S",
        "GLYCOPROTEIN":   "S",
        "NUCLEOCAPSID":   "N",
        "NUCLEOPROTEIN":  "N",
        "MEMBRANE":       "M",
        "ENVELOPE":       "E",
        "REPLICASE":      "ORF1AB",
        "POLYPROTEIN":    "ORF1AB",
        "ORF1A":          "ORF1AB",
        "ORF1B":          "ORF1AB",
        "ORF3A":          "3A",
        "ORF7A":          "7A",
        "ORF8":           "8",
        "ORF6":           "6",
        "ORF9B":          "9B",
    }

    src_nodes, dst_nodes = [], []
    matched = 0
    unmatched_sources = set()

    for _, row in meta_epi.iterrows():
        epi_idx = row["global_idx"]
        source  = seq_to_source.get(str(row["epitope_seq"]).upper(), "")

        if not source or source == "nan":
            continue

        source_upper = source.upper()
        protein_idx  = None

        # Step 1: try direct token match
        for word in source_upper.split():
            word = word.rstrip(".,;:()")
            # Try IEDB vocabulary mapping first
            canonical = IEDB_TO_GENE.get(word, word)
            if canonical in prot_token_to_idx:
                protein_idx = prot_token_to_idx[canonical]
                break
            if word in prot_token_to_idx:
                protein_idx = prot_token_to_idx[word]
                break

        # Step 2: substring match on protein_name (handles long IEDB descriptions)
        if protein_idx is None:
            for idx, row_p in meta_prot.iterrows():
                pname = str(row_p.get("protein_name", "")).upper()
                gene  = str(row_p.get("gene_name", "")).upper()
                if gene and gene in source_upper:
                    protein_idx = idx
                    break
                if pname and any(tok in source_upper for tok in pname.split()[:2]):
                    protein_idx = idx
                    break

        if protein_idx is not None:
            src_nodes.append(protein_idx)
            dst_nodes.append(int(epi_idx))
            matched += 1
        else:
            unmatched_sources.add(source[:60])

    logger.info(f"  Matched {matched:,} / {len(meta_epi):,} epitopes to source proteins")

    if unmatched_sources:
        sample = list(unmatched_sources)[:5]
        logger.debug(f"  Sample unmatched sources: {sample}")

    if not src_nodes:
        logger.warning(
            "  No protein→epitope edges built. "
            "Check that source_molecule column is populated in IEDB CSVs."
        )
        return torch.zeros((2, 0), dtype=torch.long), 0

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    return edge_index, len(src_nodes)


# ── Edge builder 2: epitope → HLA ─────────────────────────────────────────────

def build_epitope_hla_edges(data: dict) -> tuple[torch.Tensor, int]:
    """
    Edge type: (epitope, binds_to, hla)

    Two-step strategy (same as TB):
        1. Match epitopes to HLA nodes via VDJdb mhc_a allele names (exact match)
        2. Supplement with embedding similarity for epitopes with no allele match

    COVID difference from TB:
        Our HLA nodes come from VDJdb mhc_a, not a FASTA of 44,398 alleles.
        So we have only 16 HLA nodes — all of which have known allele names.
        This means step 1 (exact allele matching via VDJdb) is our primary source,
        and step 2 (similarity) is a fallback only.

    Biology: An epitope that binds HLA-A*02:01 is presented to CD8+ T-cells
    in ~50% of Caucasian populations. Knowing HLA binding breadth is critical
    for evaluating vaccine candidate coverage.
    """
    logger.info("Building epitope → HLA edges (binds_to)")

    meta_epi = data["epitope"]["meta"]
    meta_hla = data["hla"]["meta"]

    # Build: normalised allele string → HLA node index
    hla_allele_to_idx = {}
    for idx, row in meta_hla.iterrows():
        allele = str(row["allele"]).upper().strip()
        hla_allele_to_idx[allele] = idx

        # Also index without the HLA- prefix
        m = re.search(r"([A-Z0-9]+\*\d+:\d+)", allele)
        if m:
            hla_allele_to_idx[m.group(1)] = idx

    # Use VDJdb gold-filtered data for HLA matching
    # (gold-standard epitopes have confirmed allele data)
    df_vjdb_gold = data["tcr"]["df_vjdb_gold"]

    # Build: epitope_seq → set of alleles (from VDJdb)
    seq_to_alleles: dict[str, set] = {}
    if "mhc_a" in df_vjdb_gold.columns:
        for _, row in df_vjdb_gold.iterrows():
            epi = str(row.get("epitope", "")).upper().strip()
            allele = str(row.get("mhc_a", "")).upper().strip()
            if epi and allele and allele != "NAN":
                seq_to_alleles.setdefault(epi, set()).add(allele)

    epi_seq_to_idx = dict(
        zip(meta_epi["epitope_seq"].str.upper(), meta_epi["global_idx"])
    )

    src_nodes, dst_nodes = [], []

    # Step 1: exact allele name matching via VDJdb
    for epi_seq, alleles in seq_to_alleles.items():
        epi_idx = epi_seq_to_idx.get(epi_seq)
        if epi_idx is None:
            continue
        for allele in alleles:
            # Try full allele string first
            hla_idx = hla_allele_to_idx.get(allele)
            if hla_idx is None:
                # Try extracting just the allele designation
                m = re.search(r"([A-Z0-9]+\*\d+:\d+)", allele)
                if m:
                    hla_idx = hla_allele_to_idx.get(m.group(1))
            if hla_idx is not None:
                src_nodes.append(int(epi_idx))
                dst_nodes.append(int(hla_idx))

    logger.info(
        f"  Step 1 (allele name match): {len(src_nodes):,} epitope→HLA edges"
    )

    # Step 2: embedding similarity fallback for epitopes with no allele match
    matched_epi_idxs = set(src_nodes)
    n_epi = len(meta_epi)

    if len(matched_epi_idxs) < n_epi * 0.1:
        # Fewer than 10% of epitopes have explicit HLA edges — supplement
        logger.info(
            "  Step 2: supplementing with embedding similarity "
            f"(only {len(matched_epi_idxs):,} epitopes matched so far)"
        )
        emb_epi_norm = data["epitope"]["embeddings"].astype(np.float32)
        emb_hla_norm = data["hla"]["embeddings"].astype(np.float32)

        emb_epi_norm = emb_epi_norm / (
            np.linalg.norm(emb_epi_norm, axis=1, keepdims=True) + 1e-8
        )
        emb_hla_norm = emb_hla_norm / (
            np.linalg.norm(emb_hla_norm, axis=1, keepdims=True) + 1e-8
        )

        batch_size = 500
        sim_added = 0

        for i in range(0, n_epi, batch_size):
            batch = emb_epi_norm[i : i + batch_size]
            sims  = batch @ emb_hla_norm.T   # (batch, 16)
            top3  = np.argsort(sims, axis=1)[:, -3:]

            for local_idx, hla_indices in enumerate(top3):
                epi_idx = i + local_idx
                for hla_idx in hla_indices:
                    if float(sims[local_idx, hla_idx]) > 0.5:
                        src_nodes.append(epi_idx)
                        dst_nodes.append(int(hla_idx))
                        sim_added += 1

        logger.info(f"  Step 2 added: {sim_added:,} similarity-based epitope→HLA edges")

    logger.info(f"  Total epitope→HLA edges: {len(src_nodes):,}")

    if not src_nodes:
        return torch.zeros((2, 0), dtype=torch.long), 0

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    return edge_index, len(src_nodes)


# ── Edge builder 3: epitope → TCR ────────────────────────────────────────────

def build_epitope_tcr_edges(data: dict) -> tuple[torch.Tensor, int]:
    """
    Edge type: (epitope, recognized_by, tcr)

    Direct lookup from the gold-standard-filtered VDJdb data.
    Every row says "CDR3 sequence X recognises epitope Y".
    We match both to their node indices.

    COVID difference from TB:
        TB had 57 CDR3 nodes covering 11 epitopes.
        COVID has CDR3s filtered to the 668 gold-standard epitopes.
        These are the most trusted edges in the entire COVID graph —
        dual-confirmed by both immunogenicity assay (IEDB) and
        TCR sequencing (VDJdb).

    These edges are what make the COVID validation powerful:
    668 dual-confirmed epitopes vs 11 in TB means the GNN signal
    from TCR edges is ~60x richer.
    """
    logger.info("Building epitope → TCR edges (recognized_by — gold standard only)")

    meta_epi     = data["epitope"]["meta"]
    meta_tcr     = data["tcr"]["meta"]
    df_vjdb_gold = data["tcr"]["df_vjdb_gold"]

    # Build lookups
    epi_seq_to_idx = dict(
        zip(meta_epi["epitope_seq"].str.upper(), meta_epi["global_idx"])
    )
    cdr3_to_idx = dict(
        zip(meta_tcr["cdr3"].str.upper(), meta_tcr["embed_idx"])
    )

    src_nodes, dst_nodes = [], []
    skipped_no_epi  = 0
    skipped_no_cdr3 = 0

    for _, row in df_vjdb_gold.iterrows():
        epi_seq = str(row.get("epitope", "")).upper().strip()
        cdr3    = str(row.get("cdr3",    "")).upper().strip()

        epi_idx = epi_seq_to_idx.get(epi_seq)
        tcr_idx = cdr3_to_idx.get(cdr3)

        if epi_idx is None:
            skipped_no_epi += 1
            continue
        if tcr_idx is None:
            skipped_no_cdr3 += 1
            continue

        src_nodes.append(int(epi_idx))
        dst_nodes.append(int(tcr_idx))

    logger.info(
        f"  Built {len(src_nodes):,} epitope→TCR edges (gold standard)"
    )
    if skipped_no_epi > 0:
        logger.debug(
            f"  Skipped {skipped_no_epi} rows: epitope not in IEDB set"
        )
    if skipped_no_cdr3 > 0:
        logger.debug(
            f"  Skipped {skipped_no_cdr3} rows: CDR3 not in TCR node set "
            "(should be 0 since TCR nodes were built from same filtered set)"
        )

    if not src_nodes:
        logger.warning(
            "  No epitope→TCR edges. Check that VDJdb epitope sequences "
            "match IEDB sequences exactly (case, whitespace, ambiguous AA)."
        )
        return torch.zeros((2, 0), dtype=torch.long), 0

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    return edge_index, len(src_nodes)


# ── Edge builder 4: epitope ~ epitope (k-NN similarity) ──────────────────────

def build_epitope_similarity_edges(data: dict) -> tuple[torch.Tensor, int]:
    """
    Edge type: (epitope, similar_to, epitope)

    Computes pairwise cosine similarity between all epitope embeddings and
    connects each epitope to its k nearest neighbours above a threshold.

    Why this matters:
        The GNN uses this edge to propagate immunogenicity signals from
        known-positive epitopes to their similar neighbours. An epitope
        with no direct TCR or HLA evidence can still score highly if its
        sequence embedding neighbours are all immunogenic.

    Implementation: batch processing to avoid N×N memory blow-up.
        8,348 epitopes × 8,348 = ~70M pairs — fine in batches.
        (TB had 23,884 × 23,884 = 570M pairs, much harder)

    Parameters (same as TB for comparison validity):
        KNN_K = 5           — top-5 neighbours per epitope
        SIM_THRESHOLD = 0.85 — minimum cosine similarity
    """
    logger.info(
        f"Building epitope similarity edges (k={KNN_K}, threshold={SIM_THRESHOLD})"
    )

    emb   = data["epitope"]["embeddings"].astype(np.float32)
    n     = len(emb)
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    emb_norm = emb / norms

    src_nodes, dst_nodes = [], []
    batch_size = 1000

    logger.info(f"  Processing {n:,} epitopes in batches of {batch_size}...")

    for i in range(0, n, batch_size):
        batch     = emb_norm[i : i + batch_size]
        sims      = batch @ emb_norm.T              # (batch, n)
        sims_copy = sims.copy()

        # Zero out self-similarity
        np.fill_diagonal(
            sims_copy[0 : len(batch), i : i + len(batch)], 0
        )

        top_k_indices = np.argsort(sims_copy, axis=1)[:, -KNN_K:]

        for local_idx in range(len(batch)):
            global_idx = i + local_idx
            for neighbor_idx in top_k_indices[local_idx]:
                sim_val = float(sims_copy[local_idx, neighbor_idx])
                if sim_val >= SIM_THRESHOLD and int(neighbor_idx) != global_idx:
                    src_nodes.append(global_idx)
                    dst_nodes.append(int(neighbor_idx))

        if (i // batch_size) % 5 == 0:
            logger.info(
                f"  Progress: {min(i + batch_size, n):,}/{n:,} "
                f"({100 * min(i + batch_size, n) / n:.0f}%) — "
                f"{len(src_nodes):,} edges so far"
            )

    logger.info(f"  Built {len(src_nodes):,} epitope similarity edges")

    if not src_nodes:
        logger.warning(
            f"  No similarity edges at threshold={SIM_THRESHOLD}. "
            "Consider lowering SIM_THRESHOLD if COVID embeddings are more uniform."
        )
        return torch.zeros((2, 0), dtype=torch.long), 0

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    return edge_index, len(src_nodes)


# ── Assemble HeteroData ───────────────────────────────────────────────────────

def build_hetero_graph(data: dict) -> tuple[HeteroData, dict]:
    """
    Assemble all node features and edge indices into a PyG HeteroData object.

    COVID graph structure:
        graph['epitope'].x              — (8348, 320) float32
        graph['epitope'].y              — (8348,)     int64  (0/1 labels)
        graph['epitope'].seq            — list of peptide strings
        graph['epitope'].tcr_confirmed  — (8348,)     int64  (1 if gold-standard)
        graph['protein'].x              — (17,   320) float32
        graph['protein'].gene_name      — list of gene name strings
        graph['hla'].x                  — (16,   320) float32
        graph['hla'].allele             — list of allele strings
        graph['tcr'].x                  — (N,    320) float32
        graph['tcr'].cdr3               — list of CDR3 strings

    The tcr_confirmed feature on epitope nodes is an additional signal
    not present in the TB graph — it encodes the gold-standard status
    directly as a node feature, giving the GNN two ways to use this
    information: (1) via the TCR edge structure, (2) via the node feature.
    """
    console.rule("[bold cyan]Assembling COVID heterogeneous graph[/bold cyan]")

    graph = HeteroData()

    # ── Node features ─────────────────────────────────────────────────────────
    logger.info("Adding node features...")

    graph["epitope"].x = torch.tensor(
        data["epitope"]["embeddings"], dtype=torch.float32
    )
    graph["epitope"].y = torch.tensor(
        data["epitope"]["meta"]["label"].values, dtype=torch.long
    )
    graph["epitope"].seq = data["epitope"]["meta"]["epitope_seq"].tolist()

    # Gold-standard flag as node feature — unique to COVID graph
    gold_seqs = data["tcr"]["gold_standard"]
    tcr_confirmed = torch.tensor(
        [
            1 if str(s).upper() in gold_seqs else 0
            for s in data["epitope"]["meta"]["epitope_seq"]
        ],
        dtype=torch.long,
    )
    graph["epitope"].tcr_confirmed = tcr_confirmed
    n_gold_nodes = int(tcr_confirmed.sum())
    logger.info(f"  Epitopes with tcr_confirmed=1: {n_gold_nodes:,}")

    graph["protein"].x = torch.tensor(
        data["protein"]["embeddings"], dtype=torch.float32
    )
    graph["protein"].gene_name = [
        str(g) for g in data["protein"]["meta"]["gene_name"].tolist()
    ]

    graph["hla"].x = torch.tensor(
        data["hla"]["embeddings"], dtype=torch.float32
    )
    graph["hla"].allele = data["hla"]["meta"]["allele"].tolist()

    graph["tcr"].x = torch.tensor(
        data["tcr"]["embeddings"], dtype=torch.float32
    )
    graph["tcr"].cdr3 = data["tcr"]["meta"]["cdr3"].tolist()

    logger.info(f"  epitope nodes : {graph['epitope'].x.shape}")
    logger.info(f"  protein nodes : {graph['protein'].x.shape}")
    logger.info(f"  hla nodes     : {graph['hla'].x.shape}")
    logger.info(f"  tcr nodes     : {graph['tcr'].x.shape}")

    # ── Edges ─────────────────────────────────────────────────────────────────
    logger.info("Building edges...")

    ei_prot_epi, n_prot_epi = build_protein_epitope_edges(data)
    graph["protein", "source_of", "epitope"].edge_index = ei_prot_epi

    ei_epi_hla, n_epi_hla = build_epitope_hla_edges(data)
    graph["epitope", "binds_to", "hla"].edge_index = ei_epi_hla

    ei_epi_tcr, n_epi_tcr = build_epitope_tcr_edges(data)
    graph["epitope", "recognized_by", "tcr"].edge_index = ei_epi_tcr

    ei_epi_sim, n_epi_sim = build_epitope_similarity_edges(data)
    graph["epitope", "similar_to", "epitope"].edge_index = ei_epi_sim

    stats = {
        "disease":          "COVID-19 (SARS-CoV-2)",
        "n_epitopes":       data["epitope"]["n"],
        "n_positive":       int(data["epitope"]["meta"]["label"].sum()),
        "n_negative":       int((data["epitope"]["meta"]["label"] == 0).sum()),
        "n_proteins":       data["protein"]["n"],
        "n_hla":            data["hla"]["n"],
        "n_tcr":            data["tcr"]["n"],
        "n_gold_standard":  n_gold_nodes,
        "e_source_of":      n_prot_epi,
        "e_binds_to":       n_epi_hla,
        "e_recognized_by":  n_epi_tcr,
        "e_similar_to":     n_epi_sim,
        "knn_k":            KNN_K,
        "sim_threshold":    SIM_THRESHOLD,
    }

    return graph, stats


# ── Save and validate ─────────────────────────────────────────────────────────

def save_graph(graph: HeteroData, stats: dict) -> None:
    graph_path = GRAPH_DIR / "covid_graph.pt"
    torch.save(graph, str(graph_path))
    size_mb = graph_path.stat().st_size / 1e6
    logger.info(
        f"  Graph saved: {graph_path.relative_to(PROJECT_ROOT)} ({size_mb:.1f} MB)"
    )

    stats_path = GRAPH_DIR / "covid_graph_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"  Stats saved: {stats_path.relative_to(PROJECT_ROOT)}")


def validate_graph(graph: HeteroData, stats: dict) -> None:
    console.rule("[bold green]COVID Graph Validation[/bold green]")

    t = Table(
        title="COVID Heterogeneous Graph Summary",
        header_style="bold cyan", show_lines=True
    )
    t.add_column("Component",   style="white",      min_width=38)
    t.add_column("Count",       style="bold yellow", min_width=12)
    t.add_column("Details",     style="dim",          min_width=35)

    t.add_row("Epitope nodes",
              f"{stats['n_epitopes']:,}",
              f"{stats['n_positive']:,} pos / {stats['n_negative']:,} neg (1:1 balanced)")
    t.add_row("  — gold-standard (tcr_confirmed=1)",
              f"{stats['n_gold_standard']:,}",
              "IEDB pos ∩ VDJdb — dual evidence")
    t.add_row("Protein nodes",
              f"{stats['n_proteins']:,}",
              "SARS-CoV-2 reference proteome")
    t.add_row("HLA nodes",
              f"{stats['n_hla']:,}",
              "Alleles from VDJdb mhc_a")
    t.add_row("TCR nodes",
              f"{stats['n_tcr']:,}",
              "CDR3s linked to gold-standard epitopes")
    t.add_row("─" * 35, "─" * 10, "─" * 30)

    total_nodes = (
        stats["n_epitopes"] + stats["n_proteins"] +
        stats["n_hla"] + stats["n_tcr"]
    )
    t.add_row("Total nodes", f"{total_nodes:,}", "")

    t.add_row("protein → epitope edges",
              f"{stats['e_source_of']:,}",
              "source_of (Spike, N, ORF1ab...)")
    t.add_row("epitope → HLA edges",
              f"{stats['e_binds_to']:,}",
              "binds_to")
    t.add_row("epitope → TCR edges",
              f"{stats['e_recognized_by']:,}",
              "recognized_by (gold standard)")
    t.add_row("epitope → epitope edges",
              f"{stats['e_similar_to']:,}",
              f"similar_to (k={KNN_K}, t={SIM_THRESHOLD})")
    t.add_row("─" * 35, "─" * 10, "─" * 30)

    total_edges = (
        stats["e_source_of"] + stats["e_binds_to"] +
        stats["e_recognized_by"] + stats["e_similar_to"]
    )
    t.add_row("Total edges", f"{total_edges:,}", "")

    console.print(t)

    # Checks
    issues = []
    if stats["e_source_of"] == 0:
        issues.append(
            "No protein→epitope edges — source_molecule matching failed. "
            "Check if source_molecule column is populated in IEDB CSVs."
        )
    if stats["e_recognized_by"] == 0:
        issues.append(
            "No epitope→TCR edges — VDJdb matching failed. "
            "Check for case/whitespace differences in epitope sequences."
        )
    if stats["e_similar_to"] < 500:
        issues.append(
            f"Only {stats['e_similar_to']} similarity edges — "
            f"threshold {SIM_THRESHOLD} may be too high for COVID embeddings. "
            "Consider lowering SIM_THRESHOLD to 0.80."
        )
    if stats["n_tcr"] == 0:
        issues.append(
            "No TCR nodes — gold-standard filtering returned empty set. "
            "Check that IEDB and VDJdb epitope sequences match."
        )

    console.print()
    if issues:
        console.print("[bold red]Warnings:[/bold red]")
        for issue in issues:
            console.print(f"  [yellow]• {issue}[/yellow]")
    else:
        console.print("[bold green]COVID graph looks healthy![/bold green]")

    # Cross-comparison hint with TB
    console.print()
    console.print("[dim]For comparison, TB graph had:[/dim]")
    console.print("[dim]  ~23,884 epitopes, 21,008 proteins, 2,000 HLA, 57 TCR nodes[/dim]")
    console.print(
        "[dim]  COVID graph is smaller but has 60x more gold-standard TCR evidence[/dim]"
    )

    console.print(
        "\n[bold cyan]Next step:[/bold cyan] "
        "uv run python scripts/05_train_gnn_covid.py\n"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    console.rule(
        "[bold cyan]Phase 4 (COVID): Building Heterogeneous Graph[/bold cyan]"
    )
    console.print(
        "\n[bold]Framework:[/bold]  PyTorch Geometric (PyG)\n"
        "[bold]Node types:[/bold]  epitope, protein, hla, tcr\n"
        "[bold]Edge types:[/bold]  source_of, binds_to, recognized_by, similar_to\n"
        "[bold]TCR filter:[/bold]  CDR3s linked to 668 gold-standard epitopes only\n"
    )

    t0   = time.time()
    data = load_all_data()

    graph, stats = build_hetero_graph(data)
    save_graph(graph, stats)
    validate_graph(graph, stats)

    logger.info(f"  Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
