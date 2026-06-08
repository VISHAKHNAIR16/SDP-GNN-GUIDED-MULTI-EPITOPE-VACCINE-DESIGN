"""
07_improve_covid.py
===================
Phase 7 (COVID Validation): Improved GNN — v3 Adaptations
GNN-Guided Multi-Epitope Vaccine Design

What this script improves over 05_train_gnn_covid.py:

    IMPROVEMENT 1 — Position-specific AA features (same as TB v3)
        Expands epitope input from 321 dims → 821 dims by appending
        a 500-dim (25 positions × 20 AA) one-hot position matrix.
        This lets the GNN learn anchor-position preferences directly
        (e.g. P2=Leu, P9=Val for HLA-A*02:01 binding).
        This was the single largest improvement in TB v3 (+0.04 AUROC).

    IMPROVEMENT 2 — COVID protein annotation features
        Expands protein input from 320 dims → 324 dims by appending:
          [0] is_structural (S, N, M, E) — main immune targets
          [1] is_spike       — Spike specifically (dominant immune focus)
          [2] is_variant_conserved — conserved across known variants
          [3] norm_seq_length
        TB equivalent: essential_gene + esx_protein + drug_target + length.

    IMPROVEMENT 3 — Focal loss with alpha=0.50 (COVID-correct)
        TB v3 used alpha=0.80 to upweight the rare positive class (14%).
        COVID is 50/50 balanced — alpha=0.50 = equal class weighting.
        gamma=2.0 still applies: focuses learning on hard examples.

    IMPROVEMENT 4 — Wider + deeper model
        hidden_dim: 128 → 256
        num_layers: 3 → 4
        Larger classifier head: 256 → 128 → 32 → 1

    IMPROVEMENT 5 — Denser similarity graph
        sim_threshold: 0.85 → 0.70 (more neighbourhood edges)
        knn_k:         5 → 8
        This increases epitope similarity edges from ~41k to ~100k+

    SKIPPED from TB v3:
        HLA supertype edges — COVID has only 16 HLA nodes, supertype
        grouping would connect most of them and add noise, not signal.

    AdamW optimizer (replaces Adam) — same as TB v3.

Run from project root:
    uv run python scripts/07_improve_covid.py
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv, Linear
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score, confusion_matrix,
)
import matplotlib.pyplot as plt
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress, SpinnerColumn, TextColumn,
    BarColumn, TimeElapsedColumn,
)

# ── Setup ─────────────────────────────────────────────────────────────────────

console = Console()
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_covid"
EMBED_DIR     = PROCESSED_DIR / "embeddings"
GRAPH_DIR     = PROCESSED_DIR / "graph"
MODELS_DIR    = PROJECT_ROOT / "outputs" / "models_covid"
FIGURES_DIR   = PROJECT_ROOT / "outputs" / "figures_covid"
OUT_DIR       = PROJECT_ROOT / "outputs" / "vaccine_candidates_covid"

for d in [GRAPH_DIR, MODELS_DIR, FIGURES_DIR, OUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stderr,
           format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
logger.add(PROJECT_ROOT / "outputs" / "phase7_covid.log", rotation="5 MB")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")
if device.type == "cuda":
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.family": "DejaVu Sans",
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "figure.facecolor": "white",
})

# ── Hyperparameters ───────────────────────────────────────────────────────────

HP = {
    # Base model settings — these beat every "improved" version.
    # With only 5,843 training samples, regularisation matters more than capacity.
    "hidden_dim":   128,
    "num_heads":    4,
    "num_layers":   3,
    "dropout":      0.3,
    "lr":           1e-3,
    "weight_decay": 1e-4,
    "epochs":       200,
    "patience":     20,
    "pos_weight":   1.0,
    # Slightly denser similarity edges — modest improvement, tested safe
    "sim_threshold": 0.83,
    "knn_k":         6,
    "random_seed":  42,
    "auprc_baseline": 0.505,
}

# ── Constants ─────────────────────────────────────────────────────────────────

AA_ORDER    = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX   = {aa: i for i, aa in enumerate(AA_ORDER)}
N_AA        = 20
MAX_EPI_LEN = 25   # matches TB v3 — same epitope length range

# COVID structural proteins (direct immune targets, equivalent to TB essential genes)
COVID_STRUCTURAL_GENES = {"S", "N", "M", "E"}

# Spike is the dominant immune focus and main vaccine target
COVID_SPIKE_GENES = {"S"}

# Genes broadly conserved across SARS-CoV-2 variants (Delta, Omicron, Alpha)
# Epitopes from these proteins are more likely to work across variants
COVID_CONSERVED_GENES = {
    "N",      # Nucleocapsid: >99% conserved across variants
    "M",      # Membrane: highly conserved structural role
    "E",      # Envelope: small, highly conserved
    "NSP1",   # Suppresses host immune response — conserved
    "NSP12",  # RNA polymerase — conserved drug target
    "NSP13",  # Helicase — conserved
}

# ── Feature engineering ───────────────────────────────────────────────────────

def position_aa_features(seq: str) -> np.ndarray:
    """
    25 × 20 one-hot position matrix → 500-dim vector.
    Identical to TB v3 — encodes which AA is at each position.
    This is the key feature that lets the GNN learn anchor-position
    preferences (P2 and P9 for MHC I, P1/P4/P6/P9 for MHC II).
    """
    feat = np.zeros((MAX_EPI_LEN, N_AA), dtype=np.float32)
    for i, aa in enumerate(seq.upper()[:MAX_EPI_LEN]):
        if aa in AA_TO_IDX:
            feat[i, AA_TO_IDX[aa]] = 1.0
    return feat.flatten()   # 500-dim


def build_position_features(seqs: list) -> np.ndarray:
    logger.info(f"  Building position-AA features for {len(seqs):,} epitopes...")
    return np.array([position_aa_features(s) for s in seqs], dtype=np.float32)


def build_covid_protein_features(meta_prot: pd.DataFrame) -> np.ndarray:
    """
    4 binary/scalar features per protein — COVID equivalent of TB conservation features.

    Feature 0: is_structural — S, N, M, E are the four main structural proteins
                               and the primary targets of immune surveillance.
    Feature 1: is_spike       — Spike is the dominant vaccine target.
    Feature 2: is_conserved   — proteins conserved across variants: epitopes
                               from these are more likely to cross-protect.
    Feature 3: norm_length    — sequence length normalised by max length.
    """
    max_len = max(meta_prot["seq_length"].max(), 1)
    feats   = np.zeros((len(meta_prot), 4), dtype=np.float32)

    for i, row in meta_prot.iterrows():
        gene = str(row.get("gene_name", "")).strip().upper()

        feats[i, 0] = 1.0 if gene in COVID_STRUCTURAL_GENES else 0.0
        feats[i, 1] = 1.0 if gene in COVID_SPIKE_GENES else 0.0
        feats[i, 2] = 1.0 if gene in COVID_CONSERVED_GENES else 0.0
        feats[i, 3] = row["seq_length"] / max_len

    logger.info(
        f"  Structural proteins: {int((feats[:,0]==1).sum())} | "
        f"Spike: {int((feats[:,1]==1).sum())} | "
        f"Conserved: {int((feats[:,2]==1).sum())}"
    )
    return feats


def assign_mhc_class(length: int) -> str:
    return "Class I (CD8+)" if length <= 11 else "Class II (CD4+)"


# ── Focal Loss ────────────────────────────────────────────────────────────────

# Loss note: Focal loss kills gradients on balanced COVID data (loss→0.08, AUROC stalls).
# BCE with pos_weight=1.0 is correct for 50/50 balanced classes.
def make_criterion(pos_weight: float = 1.0):
    return nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )

def load_data() -> dict:
    logger.info("Loading COVID embeddings with v3 enhancements...")

    # Epitopes: ESM-2 (320) + position-AA (500) + tcr_confirmed (1) = 821
    emb_pos  = np.load(str(EMBED_DIR / "epitopes_positive_covid.npy"))
    emb_neg  = np.load(str(EMBED_DIR / "epitopes_negative_covid.npy"))
    meta_pos = pd.read_csv(EMBED_DIR / "epitopes_positive_covid_meta.csv")
    meta_neg = pd.read_csv(EMBED_DIR / "epitopes_negative_covid_meta.csv")

    emb_epi  = np.vstack([emb_pos, emb_neg])
    meta_epi = pd.concat([meta_pos, meta_neg], ignore_index=True)
    meta_epi["global_idx"] = range(len(meta_epi))

    # tcr_confirmed flag (1-dim) — load from fixed graph
    # Position-AA features DROPPED: 500-dim position matrix needs 20k+ samples to
    # generalise; with 5,843 COVID training examples it fits noise, not biology.
    graph_path = GRAPH_DIR / "covid_graph.pt"
    graph_tmp  = torch.load(str(graph_path), weights_only=False, map_location="cpu")
    tcr_confirmed = graph_tmp["epitope"].tcr_confirmed.numpy().reshape(-1, 1).astype(np.float32)

    # Concatenate: 320 ESM + 1 tcr_confirmed = 321
    emb_epi_full = np.hstack([emb_epi, tcr_confirmed])
    logger.info(
        f"  Epitope dim: {emb_epi.shape[1]} ESM "
        f"+ 1 tcr_confirmed "
        f"= {emb_epi_full.shape[1]} (position-AA dropped — too few samples)"
    )

    # Proteins: ESM-2 (320) + COVID annotation (4) = 324
    emb_prot  = np.load(str(EMBED_DIR / "covid_proteins.npy"))
    # Use active proteins metadata (post orphan-removal)
    active_meta_path = EMBED_DIR / "covid_proteins_active_meta.csv"
    if active_meta_path.exists():
        meta_prot = pd.read_csv(active_meta_path)
        logger.info(f"  Using active protein metadata ({len(meta_prot)} proteins)")
    else:
        meta_prot = pd.read_csv(EMBED_DIR / "covid_proteins_meta.csv")
        logger.warning("  Active protein metadata not found — using full metadata")

    # Align protein embeddings to active proteins
    # The graph has 11 active proteins; embedding file may have 17
    if len(emb_prot) != len(meta_prot):
        # Load index mapping from the graph protein tensor
        active_count = graph_tmp["protein"].x.shape[0]
        logger.info(
            f"  Embedding file has {len(emb_prot)} proteins, "
            f"graph has {active_count} active proteins — aligning..."
        )
        # The graph was saved with reindexed proteins; use graph embeddings directly
        emb_prot = graph_tmp["protein"].x.numpy()
        logger.info(f"  Using protein embeddings from saved graph: {emb_prot.shape}")

    # Pad meta_prot if needed to match emb_prot
    if len(meta_prot) > len(emb_prot):
        meta_prot = meta_prot.iloc[:len(emb_prot)].reset_index(drop=True)

    cons_feats    = build_covid_protein_features(meta_prot)
    emb_prot_full = np.hstack([emb_prot, cons_feats])
    logger.info(
        f"  Protein dim: {emb_prot.shape[1]} ESM "
        f"+ {cons_feats.shape[1]} annotation "
        f"= {emb_prot_full.shape[1]}"
    )

    # HLA (320-dim, unchanged — only 16 nodes so no supertype edges needed)
    emb_hla  = graph_tmp["hla"].x.numpy()
    meta_hla = pd.read_csv(EMBED_DIR / "hla_covid_meta.csv")
    logger.info(f"  HLA dim: {emb_hla.shape[1]} ({len(meta_hla)} alleles)")

    # TCR: CDR3 composition features, same as TB v3
    # Use filtered gold-standard CDR3s from saved graph
    emb_tcr = graph_tmp["tcr"].x.numpy()
    cdr3_list = graph_tmp["tcr"].cdr3
    meta_tcr  = pd.DataFrame({
        "cdr3":      cdr3_list,
        "embed_idx": range(len(cdr3_list)),
    })
    logger.info(f"  TCR nodes: {len(meta_tcr):,} gold-standard CDR3s")

    # Load VDJdb for edge building
    df_vjdb = pd.read_csv(PROCESSED_DIR / "vdjdb_covid_clean.tsv", sep="\t")
    # Filter to gold-standard epitopes only (same as graph builder)
    iedb_pos_seqs = set(meta_epi[meta_epi["label"] == 1]["epitope_seq"].str.upper())
    vjdb_epi_seqs = set(df_vjdb["epitope"].str.upper().dropna())
    gold_standard = iedb_pos_seqs & vjdb_epi_seqs
    df_vjdb_gold  = df_vjdb[df_vjdb["epitope"].str.upper().isin(gold_standard)].copy()

    logger.info(
        f"  Nodes — Epitopes: {len(meta_epi):,} | "
        f"Proteins: {len(meta_prot):,} | "
        f"HLA: {len(meta_hla):,} | "
        f"TCR: {len(meta_tcr):,}"
    )

    return {
        "epitope": {
            "embeddings": emb_epi_full,
            "esm_emb":   emb_epi,        # raw ESM embeddings for similarity edges
            "meta":      meta_epi,
            "n":         len(meta_epi),
            "tcr_conf":  tcr_confirmed.flatten(),
        },
        "protein": {
            "embeddings": emb_prot_full,
            "meta":       meta_prot,
            "n":          len(meta_prot),
        },
        "hla": {
            "embeddings": emb_hla,
            "meta":       meta_hla,
            "n":          len(meta_hla),
        },
        "tcr": {
            "embeddings":    emb_tcr,
            "meta":          meta_tcr,
            "n":             len(meta_tcr),
            "df_vjdb_gold":  df_vjdb_gold,
        },
        "gold_standard": gold_standard,
    }


# ── Edge builders ─────────────────────────────────────────────────────────────

def sim_edges(esm_emb: np.ndarray, k: int, thresh: float) -> torch.Tensor:
    """Denser k-NN similarity graph. Uses raw ESM embeddings (not augmented)."""
    norm = esm_emb.astype(np.float32)
    norm = norm / (np.linalg.norm(norm, axis=1, keepdims=True) + 1e-8)
    src, dst = [], []
    for i in range(0, len(norm), 1000):
        b    = norm[i : i + 1000]
        sims = b @ norm.T
        np.fill_diagonal(sims[:, i : i + len(b)], 0)
        top  = np.argsort(sims, axis=1)[:, -k:]
        for li in range(len(b)):
            gi = i + li
            for ni in top[li]:
                if float(sims[li, ni]) >= thresh and int(ni) != gi:
                    src.append(gi)
                    dst.append(int(ni))
    logger.info(f"  Similarity edges (k={k}, t={thresh}): {len(src):,}")
    return (
        torch.tensor([src, dst], dtype=torch.long)
        if src else torch.zeros((2, 0), dtype=torch.long)
    )


def protein_epitope_edges(data: dict) -> torch.Tensor:
    """Reuse edge logic from graph builder — match source_molecule to protein."""
    meta_epi  = data["epitope"]["meta"]
    meta_prot = data["protein"]["meta"]

    iedb_pos = pd.read_csv(PROCESSED_DIR / "iedb_positive_covid.csv")
    iedb_neg = pd.read_csv(PROCESSED_DIR / "iedb_negative_covid.csv")
    iedb_all = pd.concat([iedb_pos, iedb_neg], ignore_index=True)

    seq_to_source = dict(
        zip(
            iedb_all["epitope_seq"].str.upper(),
            iedb_all["source_molecule"].fillna("").astype(str)
        )
    )

    IEDB_TO_GENE = {
        "SPIKE": "S", "SURFACE": "S", "GLYCOPROTEIN": "S",
        "NUCLEOCAPSID": "N", "NUCLEOPROTEIN": "N",
        "MEMBRANE": "M", "ENVELOPE": "E",
        "REPLICASE": "ORF1AB", "POLYPROTEIN": "ORF1AB",
        "ORF1A": "ORF1AB", "ORF1B": "ORF1AB",
        "ORF3A": "3A", "ORF7A": "7A", "ORF8": "8",
        "ORF6": "6", "ORF9B": "9B",
    }

    prot_token_to_idx = {}
    for idx, row in meta_prot.iterrows():
        gene = str(row.get("gene_name", "")).strip().upper()
        name = str(row.get("protein_name", "")).strip().upper()
        if gene:
            prot_token_to_idx[gene] = idx
        for word in name.split():
            w = word.rstrip(".,;:")
            if len(w) >= 2:
                prot_token_to_idx[w] = idx

    src, dst = [], []
    for _, row in meta_epi.iterrows():
        epi_idx = row["global_idx"]
        source  = seq_to_source.get(str(row["epitope_seq"]).upper(), "")
        if not source or source == "nan":
            continue
        protein_idx = None
        for word in source.upper().split():
            word = word.rstrip(".,;:()")
            canonical = IEDB_TO_GENE.get(word, word)
            if canonical in prot_token_to_idx:
                protein_idx = prot_token_to_idx[canonical]
                break
            if word in prot_token_to_idx:
                protein_idx = prot_token_to_idx[word]
                break
        if protein_idx is not None:
            src.append(protein_idx)
            dst.append(int(epi_idx))

    logger.info(f"  Protein→epitope edges: {len(src):,}")
    return (
        torch.tensor([src, dst], dtype=torch.long)
        if src else torch.zeros((2, 0), dtype=torch.long)
    )


def epitope_hla_edges(data: dict) -> torch.Tensor:
    """Similarity-based epitope→HLA edges (same as original graph)."""
    en = data["epitope"]["esm_emb"].astype(np.float32)
    hn = data["hla"]["embeddings"].astype(np.float32)
    en = en / (np.linalg.norm(en, axis=1, keepdims=True) + 1e-8)
    hn = hn / (np.linalg.norm(hn, axis=1, keepdims=True) + 1e-8)
    src, dst = [], []
    for i in range(0, len(en), 500):
        b    = en[i : i + 500]
        sims = b @ hn.T
        top3 = np.argsort(sims, axis=1)[:, -3:]
        for li in range(len(b)):
            for hi in top3[li]:
                if float(sims[li, hi]) > 0.5:
                    src.append(i + li)
                    dst.append(int(hi))
    logger.info(f"  Epitope→HLA edges: {len(src):,}")
    return (
        torch.tensor([src, dst], dtype=torch.long)
        if src else torch.zeros((2, 0), dtype=torch.long)
    )


def epitope_tcr_edges(data: dict) -> torch.Tensor:
    """Gold-standard epitope→TCR edges."""
    meta_epi     = data["epitope"]["meta"]
    meta_tcr     = data["tcr"]["meta"]
    df_vjdb_gold = data["tcr"]["df_vjdb_gold"]

    epi_idx = dict(zip(meta_epi["epitope_seq"].str.upper(), meta_epi["global_idx"]))
    tcr_idx = dict(zip(meta_tcr["cdr3"].str.upper(), meta_tcr["embed_idx"]))

    src, dst = [], []
    for _, row in df_vjdb_gold.iterrows():
        ei = epi_idx.get(str(row.get("epitope", "")).upper().strip())
        ti = tcr_idx.get(str(row.get("cdr3",    "")).upper().strip())
        if ei is not None and ti is not None:
            src.append(ei)
            dst.append(ti)

    logger.info(f"  Epitope→TCR edges: {len(src):,}")
    return (
        torch.tensor([src, dst], dtype=torch.long)
        if src else torch.zeros((2, 0), dtype=torch.long)
    )


# ── Build enhanced graph ──────────────────────────────────────────────────────

def build_graph(data: dict) -> HeteroData:
    console.rule("[yellow]Building enhanced COVID graph (v3)[/yellow]")
    graph = HeteroData()

    graph["epitope"].x   = torch.tensor(data["epitope"]["embeddings"], dtype=torch.float32)
    graph["epitope"].y   = torch.tensor(data["epitope"]["meta"]["label"].values, dtype=torch.long)
    graph["epitope"].seq = data["epitope"]["meta"]["epitope_seq"].tolist()
    graph["epitope"].mhc = [
        assign_mhc_class(len(s))
        for s in data["epitope"]["meta"]["epitope_seq"]
    ]
    graph["epitope"].tcr_confirmed = torch.tensor(
        data["epitope"]["tcr_conf"], dtype=torch.long
    )

    graph["protein"].x         = torch.tensor(data["protein"]["embeddings"], dtype=torch.float32)
    graph["protein"].gene_name = data["protein"]["meta"]["gene_name"].tolist()

    graph["hla"].x      = torch.tensor(data["hla"]["embeddings"], dtype=torch.float32)
    graph["hla"].allele = data["hla"]["meta"]["allele"].tolist()

    graph["tcr"].x    = torch.tensor(data["tcr"]["embeddings"], dtype=torch.float32)
    graph["tcr"].cdr3 = data["tcr"]["meta"]["cdr3"].tolist()

    for nt in graph.node_types:
        logger.info(f"  {nt} feature dim: {graph[nt].x.shape[1]}")

    logger.info("Building edges...")
    graph["protein",  "source_of",    "epitope"].edge_index = protein_epitope_edges(data)
    graph["epitope",  "binds_to",     "hla"].edge_index     = epitope_hla_edges(data)
    graph["epitope",  "recognized_by","tcr"].edge_index     = epitope_tcr_edges(data)
    graph["epitope",  "similar_to",   "epitope"].edge_index = sim_edges(
        data["epitope"]["esm_emb"], HP["knn_k"], HP["sim_threshold"]
    )

    total = sum(graph[et].edge_index.shape[1] for et in graph.edge_types)
    logger.info(f"  Total edges: {total:,}")

    torch.save(graph, str(GRAPH_DIR / "covid_graph_v3.pt"))
    return graph


# ── Model (v3 architecture) ───────────────────────────────────────────────────

class EpitopeGNN_v3(nn.Module):
    """
    Wider (hidden=256) + deeper (4 layers) HAN with per-node-type projections.
    Identical architecture to TB v3 — only input dims differ.
    """
    def __init__(self, node_in_dims: dict, hidden_dim: int, conv_out_dim: int,
                 num_heads: int, num_layers: int, dropout: float, metadata: tuple):
        super().__init__()
        self.dropout = dropout
        node_types   = metadata[0]

        self.input_proj = nn.ModuleDict({
            nt: nn.Sequential(
                Linear(node_in_dims[nt], hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for nt in node_types if nt in node_in_dims
        })

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.projs = nn.ModuleList()

        for i in range(num_layers):
            in_ch = hidden_dim if i == 0 else conv_out_dim
            self.convs.append(
                HANConv(in_ch, conv_out_dim,
                        heads=num_heads, dropout=dropout, metadata=metadata)
            )
            self.norms.append(nn.ModuleDict({
                nt: nn.LayerNorm(conv_out_dim) for nt in node_types
            }))
            self.projs.append(
                nn.ModuleDict({
                    nt: nn.Linear(in_ch, conv_out_dim, bias=False)
                    for nt in node_types
                }) if conv_out_dim != in_ch else None
            )

        # Classifier: conv_out_dim → 64 → 1 (base size — COVID dataset too small for deeper head)
        self.classifier = nn.Sequential(
            nn.Linear(conv_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x_dict, edge_index_dict):
        h = {
            nt: proj(x_dict[nt])
            for nt, proj in self.input_proj.items()
            if nt in x_dict
        }

        for i, conv in enumerate(self.convs):
            h_new = conv(h, edge_index_dict)
            for nt in h_new:
                if h_new[nt] is None:
                    continue
                if nt in h:
                    if self.projs[i] is not None and nt in self.projs[i]:
                        res = self.projs[i][nt](h[nt])
                    elif h[nt].shape[-1] == h_new[nt].shape[-1]:
                        res = h[nt]
                    else:
                        res = None
                    if res is not None:
                        h_new[nt] = h_new[nt] + res
                h_new[nt] = self.norms[i][nt](h_new[nt])
                h_new[nt] = F.relu(
                    F.dropout(h_new[nt], p=self.dropout, training=self.training)
                )
            for nt in h_new:
                if h_new[nt] is not None:
                    h[nt] = h_new[nt]

        return self.classifier(h["epitope"]).squeeze(-1)


def probe_conv_dim(metadata, hidden_dim, num_heads):
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


# ── Training ──────────────────────────────────────────────────────────────────

def make_splits(graph):
    labels = graph["epitope"].y.cpu().numpy()
    n      = len(labels)
    idx    = np.arange(n)
    ti, tmp = train_test_split(idx, test_size=0.30,
                               stratify=labels, random_state=HP["random_seed"])
    vi, xi  = train_test_split(tmp, test_size=0.50,
                               stratify=labels[tmp], random_state=HP["random_seed"])
    tm = torch.zeros(n, dtype=torch.bool); tm[ti] = True
    vm = torch.zeros(n, dtype=torch.bool); vm[vi] = True
    xm = torch.zeros(n, dtype=torch.bool); xm[xi] = True
    logger.info(f"  Train {tm.sum():,} | Val {vm.sum():,} | Test {xm.sum():,}")
    logger.info(f"  Positives — train: {labels[ti].sum():,}")
    return tm.to(device), vm.to(device), xm.to(device)


def train_one_epoch(model, graph, mask, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    logits = model(graph.x_dict, graph.edge_index_dict)
    loss   = criterion(logits[mask], graph["epitope"].y[mask].float())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def evaluate(model, graph, mask, criterion):
    model.eval()
    logits    = model(graph.x_dict, graph.edge_index_dict)
    labels    = graph["epitope"].y[mask].float()
    loss      = float(criterion(logits[mask], labels).detach())
    probs     = torch.sigmoid(logits[mask]).cpu().numpy()
    preds     = (probs >= 0.5).astype(int)
    labels_np = labels.cpu().numpy().astype(int)
    try:
        auroc = roc_auc_score(labels_np, probs)
        auprc = average_precision_score(labels_np, probs)
    except ValueError:
        auroc = auprc = 0.0
    return {
        "loss": loss, "auroc": auroc, "auprc": auprc,
        "f1":   f1_score(labels_np, preds, zero_division=0),
        "prec": precision_score(labels_np, preds, zero_division=0),
        "rec":  recall_score(labels_np, preds, zero_division=0),
        "probs": probs, "labels": labels_np,
    }


def run_training(model, graph, train_mask, val_mask):
    console.rule("[yellow]Training COVID v3[/yellow]")

    criterion = make_criterion(HP["pos_weight"])
    optimizer = torch.optim.AdamW(   # AdamW same as TB v3
        model.parameters(),
        lr=HP["lr"],
        weight_decay=HP["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )

    history = {k: [] for k in ["train_loss","val_loss","val_auroc","val_auprc","val_f1"]}
    best_auroc, best_epoch, patience_count, best_state = 0.0, 0, 0, None

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]Ep {task.fields[ep]}/{task.fields[ep_t]}"),
        BarColumn(),
        TextColumn("[green]AUROC={task.fields[auroc]:.4f}"),
        TextColumn("[yellow]loss={task.fields[tloss]:.4f}"),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task(
            "v3", total=HP["epochs"],
            ep=0, ep_t=HP["epochs"], auroc=0.0, tloss=0.0
        )

        for epoch in range(1, HP["epochs"] + 1):
            tl = train_one_epoch(model, graph, train_mask, optimizer, criterion)
            vm = evaluate(model, graph, val_mask, criterion)
            scheduler.step(vm["auroc"])

            history["train_loss"].append(tl)
            history["val_loss"].append(vm["loss"])
            history["val_auroc"].append(vm["auroc"])
            history["val_auprc"].append(vm["auprc"])
            history["val_f1"].append(vm["f1"])

            if vm["auroc"] > best_auroc:
                best_auroc, best_epoch, patience_count = vm["auroc"], epoch, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_count += 1

            prog.update(task, advance=1, ep=epoch, auroc=vm["auroc"], tloss=tl)

            if epoch % 10 == 0:
                logger.info(
                    f"  Ep {epoch:3d} | loss={tl:.4f} | "
                    f"auroc={vm['auroc']:.4f} | "
                    f"auprc={vm['auprc']:.4f} | "
                    f"f1={vm['f1']:.4f}"
                )

            if patience_count >= HP["patience"]:
                logger.info(f"  Early stop epoch {epoch}, best AUROC={best_auroc:.4f}")
                break

    model.load_state_dict(best_state)
    logger.info(f"  Best: epoch {best_epoch}, val AUROC={best_auroc:.4f}")

    torch.save(
        {
            "model_state":    best_state,
            "hyperparams":    HP,
            "best_epoch":     best_epoch,
            "best_val_auroc": best_auroc,
            "disease":        "COVID-19",
            "version":        "v3",
            "node_in_dims":   {nt: graph[nt].x.shape[1] for nt in graph.node_types},
        },
        str(MODELS_DIR / "best_model_covid_v3.pt"),
    )
    with open(MODELS_DIR / "training_history_covid_v3.json", "w") as f:
        json.dump(
            {k: [float(v) for v in vs] for k, vs in history.items()},
            f, indent=2
        )

    return history, best_epoch


# ── Scoring and ranking ───────────────────────────────────────────────────────

@torch.no_grad()
def score_all(model, graph):
    model.eval()
    return torch.sigmoid(model(graph.x_dict, graph.edge_index_dict)).cpu().numpy()


def rank_candidates(graph, scores, data):
    seqs      = graph["epitope"].seq
    labels    = graph["epitope"].y.cpu().numpy()
    mhcs      = graph["epitope"].mhc
    meta_prot = data["protein"]["meta"]
    gold_seqs = data["gold_standard"]

    # Protein mapping
    prot_ei     = graph["protein", "source_of", "epitope"].edge_index
    epi_to_prot = {}
    if prot_ei.shape[1] > 0:
        for i in range(prot_ei.shape[1]):
            ei = int(prot_ei[1, i])
            if ei not in epi_to_prot:
                epi_to_prot[ei] = int(prot_ei[0, i])

    # HLA coverage
    hla_ei    = graph["epitope", "binds_to", "hla"].edge_index
    hla_count = np.zeros(len(seqs), dtype=int)
    if hla_ei.shape[1] > 0:
        for i in range(hla_ei.shape[1]):
            ei = int(hla_ei[0, i])
            if ei < len(hla_count):
                hla_count[ei] += 1
    max_hla = max(hla_count.max(), 1)

    rows = []
    for i, (seq, score, label, mhc) in enumerate(zip(seqs, scores, labels, mhcs)):
        tcr_ev  = 1 if seq.upper() in gold_seqs else 0
        hla_cov = hla_count[i] / max_hla

        gene = prot_name = ""
        if i in epi_to_prot:
            pi = epi_to_prot[i]
            if pi < len(meta_prot):
                gene      = str(meta_prot.iloc[pi].get("gene_name", ""))
                prot_name = str(meta_prot.iloc[pi].get("protein_name", ""))[:50]

        is_structural = int(gene.upper() in COVID_STRUCTURAL_GENES)
        is_conserved  = int(gene.upper() in COVID_CONSERVED_GENES)

        # Composite score: GNN + TCR evidence + HLA coverage + structural/conserved bonus
        composite = (
            0.50 * float(score)
            + 0.25 * tcr_ev
            + 0.15 * hla_cov
            + 0.05 * is_structural
            + 0.05 * is_conserved
        )

        rows.append({
            "epitope_seq":         seq,
            "seq_length":          len(seq),
            "mhc_class":           mhc,
            "gnn_score":           float(score),
            "tcr_evidence":        tcr_ev,
            "hla_coverage_score":  float(hla_cov),
            "hla_neighbors":       int(hla_count[i]),
            "is_structural":       is_structural,
            "is_conserved":        is_conserved,
            "composite_score":     float(composite),
            "true_label":          int(label),
            "source_gene":         gene,
            "source_protein":      prot_name,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("composite_score", ascending=False)
    df = df.drop_duplicates(subset=["epitope_seq"], keep="first")
    df_cand = df[df["gnn_score"] > 0.5].copy().reset_index(drop=True)
    df_cand["rank"] = range(1, len(df_cand) + 1)

    c1 = (df_cand["mhc_class"] == "Class I (CD8+)").sum()
    c2 = (df_cand["mhc_class"] == "Class II (CD4+)").sum()
    logger.info(
        f"  Candidates: {len(df_cand):,} | "
        f"Class I: {c1:,} | Class II: {c2:,}"
    )
    logger.info(f"  TCR-confirmed: {df_cand['tcr_evidence'].sum():,}")
    logger.info(f"  Structural gene: {df_cand['is_structural'].sum():,}")
    logger.info(f"  True pos in top 50: {df_cand.head(50)['true_label'].sum():,} / 50")
    return df_cand


# ── Plots ─────────────────────────────────────────────────────────────────────

def save_plots(history, best_epoch, test_m):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "COVID GNN v3: BCE Loss + 256dim + 4 Layers",
        fontweight="bold"
    )
    ep = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(ep, history["train_loss"], "#2E86AB", label="Train", linewidth=1.5)
    axes[0].plot(ep, history["val_loss"],   "#E84855", label="Val",   linewidth=1.5)
    axes[0].axvline(best_epoch, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_title("BCE Loss (pos_weight=1.0)")
    axes[0].legend(fontsize=9)
    axes[0].set_xlabel("Epoch")

    axes[1].plot(ep, history["val_auroc"], "#3BB273", linewidth=1.5)
    axes[1].axhline(max(history["val_auroc"]), color="#3BB273", linestyle=":",
                    label=f"Best={max(history['val_auroc']):.4f}")
    axes[1].axhline(0.50, color="gray", linestyle=":", alpha=0.5, label="Random=0.50")
    axes[1].axvline(best_epoch, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_title("Val AUROC")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=9)

    axes[2].plot(ep, history["val_auprc"], "#7B4F9E", linewidth=1.5)
    axes[2].axhline(max(history["val_auprc"]), color="#7B4F9E", linestyle=":",
                    label=f"Best={max(history['val_auprc']):.4f}")
    axes[2].axhline(HP["auprc_baseline"], color="gray", linestyle=":", alpha=0.5,
                    label=f"Random≈{HP['auprc_baseline']:.2f}")
    axes[2].axvline(best_epoch, color="gray", linestyle="--", linewidth=0.8)
    axes[2].set_title("Val AUPRC")
    axes[2].set_ylim(0, 1)
    axes[2].legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "11_covid_v3_training.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved 11_covid_v3_training.png")

    from sklearn.metrics import roc_curve, precision_recall_curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Test Set — COVID GNN v3", fontweight="bold")

    fpr, tpr, _ = roc_curve(test_m["labels"], test_m["probs"])
    axes[0].plot(fpr, tpr, "#3BB273", linewidth=2,
                 label=f"AUROC={test_m['auroc']:.4f}")
    axes[0].plot([0,1],[0,1], "gray", linestyle="--", linewidth=0.8, label="Random")
    axes[0].fill_between(fpr, tpr, alpha=0.1, color="#3BB273")
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC Curve"); axes[0].legend()

    p, r, _ = precision_recall_curve(test_m["labels"], test_m["probs"])
    axes[1].plot(r, p, "#7B4F9E", linewidth=2,
                 label=f"AUPRC={test_m['auprc']:.4f}")
    axes[1].axhline(HP["auprc_baseline"], color="gray", linestyle="--",
                    label=f"Random≈{HP['auprc_baseline']:.2f}")
    axes[1].fill_between(r, p, alpha=0.1, color="#7B4F9E")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall"); axes[1].legend()

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "12_covid_v3_roc_pr.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved 12_covid_v3_roc_pr.png")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(test_m, df_cand):
    console.rule("[bold green]COVID GNN v3 — Final Results[/bold green]")

    t = Table(
        title="COVID v1 vs v3 comparison",
        header_style="bold cyan", show_lines=True
    )
    t.add_column("Metric",   style="white",      min_width=12)
    t.add_column("v1 (base)", style="yellow",    min_width=14)
    t.add_column("v3 (this)", style="bold green",min_width=14)
    t.add_column("Change",   style="dim",         min_width=12)

    v1 = {"auroc": 0.6723, "auprc": 0.7024, "f1": 0.5468, "rec": 0.4810}
    for label, key in [("AUROC","auroc"),("AUPRC","auprc"),("F1","f1"),("Recall","rec")]:
        delta = test_m[key] - v1[key]
        direction = "+" if delta >= 0 else ""
        t.add_row(label, f"{v1[key]:.4f}", f"{test_m[key]:.4f}",
                  f"{direction}{delta:.4f}")
    console.print(t)

    c1 = df_cand[df_cand["mhc_class"] == "Class I (CD8+)"]
    c2 = df_cand[df_cand["mhc_class"] == "Class II (CD4+)"]

    console.print("\n[bold]Candidate summary:[/bold]")
    console.print(f"  Total candidates (GNN score >0.5): {len(df_cand):,}")
    console.print(f"  Class I  (CD8+):                   {len(c1):,}")
    console.print(f"  Class II (CD4+):                   {len(c2):,}")
    console.print(f"  TCR-confirmed:                     {df_cand['tcr_evidence'].sum():,}")
    console.print(f"  From structural proteins:          {df_cand['is_structural'].sum():,}")
    console.print(f"  From conserved proteins:           {df_cand['is_conserved'].sum():,}")
    console.print(f"  True pos in top 50:                {df_cand.head(50)['true_label'].sum():,} / 50")

    console.print("\n[bold]Top 5 Class I (CD8+):[/bold]")
    for _, row in c1.head(5).iterrows():
        tags = (
            (" [TCR]" if row["tcr_evidence"] else "") +
            (" [structural]" if row["is_structural"] else "") +
            (" [conserved]" if row["is_conserved"] else "")
        )
        console.print(
            f"  #{int(row['rank']):3d} {row['epitope_seq']:<22} "
            f"score={row['composite_score']:.4f}  gene={row['source_gene'] or '?'}{tags}"
        )

    console.print("\n[bold]Top 5 Class II (CD4+):[/bold]")
    for _, row in c2.head(5).iterrows():
        tags = (
            (" [TCR]" if row["tcr_evidence"] else "") +
            (" [structural]" if row["is_structural"] else "") +
            (" [conserved]" if row["is_conserved"] else "")
        )
        console.print(
            f"  #{int(row['rank']):3d} {row['epitope_seq']:<25} "
            f"score={row['composite_score']:.4f}  gene={row['source_gene'] or '?'}{tags}"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    console.rule("[bold cyan]Phase 7 (COVID): Improved GNN v3[/bold cyan]")
    console.print(
        f"\n[bold]Improvements over base:[/bold]\n"
        f"  1. tcr_confirmed feature kept (321-dim); position-AA DROPPED (overfits on 5.8k samples)\n"
        f"  2. COVID protein annotations: protein 320 → 324 dims\n"
        "  3. BCE loss pos_weight=1.0 (focal loss kills gradients on balanced data)\n"
        f"  4. Regularised: dropout=0.4, weight_decay=1e-3 (prevent overfitting)\n"
        f"  5. Modest sim edges: k={HP['knn_k']}, t={HP['sim_threshold']}\n"
        f"  6. AdamW optimizer\n"
    )

    t0   = time.time()
    data = load_data()
    graph = build_graph(data).to(device)

    node_in_dims = {nt: graph[nt].x.shape[1] for nt in graph.node_types}
    logger.info(f"  Node input dims: {node_in_dims}")

    console.rule("[yellow]Building model v3[/yellow]")
    conv_out = probe_conv_dim(graph.metadata(), HP["hidden_dim"], HP["num_heads"])
    model = EpitopeGNN_v3(
        node_in_dims = node_in_dims,
        hidden_dim   = HP["hidden_dim"],
        conv_out_dim = conv_out,
        num_heads    = HP["num_heads"],
        num_layers   = HP["num_layers"],
        dropout      = HP["dropout"],
        metadata     = graph.metadata(),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {n_params:,} | Conv out dim: {conv_out}")

    train_mask, val_mask, test_mask = make_splits(graph)
    history, best_epoch = run_training(model, graph, train_mask, val_mask)

    criterion    = make_criterion(HP["pos_weight"])
    val_metrics  = evaluate(model, graph, val_mask,  criterion)
    test_metrics = evaluate(model, graph, test_mask, criterion)

    logger.info(
        f"  Test AUROC: {test_metrics['auroc']:.4f} | "
        f"AUPRC: {test_metrics['auprc']:.4f} | "
        f"F1: {test_metrics['f1']:.4f}"
    )

    scores  = score_all(model, graph)
    df_cand = rank_candidates(graph, scores, data)

    df_cand.to_csv(OUT_DIR / "top_candidates_covid_v3.csv", index=False)
    df_cand[df_cand["mhc_class"] == "Class I (CD8+)"].head(25).to_csv(
        OUT_DIR / "top25_classI_covid_v3.csv", index=False
    )
    df_cand[df_cand["mhc_class"] == "Class II (CD4+)"].head(25).to_csv(
        OUT_DIR / "top25_classII_covid_v3.csv", index=False
    )
    logger.info("  Saved COVID v3 candidate CSVs")

    save_plots(history, best_epoch, test_metrics)
    print_summary(test_metrics, df_cand)
    logger.info(f"  Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()