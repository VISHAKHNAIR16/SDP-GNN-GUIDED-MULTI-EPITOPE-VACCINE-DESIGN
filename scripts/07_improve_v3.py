"""
07_improve_v3.py
================
Phase 7 v3: Maximum GNN Improvement — FINAL FIXED VERSION
GNN-Guided Multi-Epitope Vaccine Design

Fixes from failed v3:
    FIX 1: focal_alpha 0.25 → 0.80
            Alpha=0.25 meant negatives got 75% weight — completely wrong for
            9.6:1 imbalance. Alpha=0.80 gives 80% weight to the minority
            positive class, which is what we actually want.

    FIX 2: CosineAnnealingLR → ReduceLROnPlateau
            Cosine annealing drove LR to near-zero by epoch 200, stopping
            all learning. ReduceLROnPlateau only reduces LR when val AUROC
            stalls, which is the correct behaviour.

    FIX 3: Removed layer-wise learning rates
            The 0.3x LR on input_proj meant the new 820-dim feature extraction
            layers (the most important new part) barely updated. Uniform LR
            for all layers.

    Everything else from v3 is kept:
        - Focal loss (now correctly configured)
        - hidden_dim=256, num_layers=4
        - Per-node-type input projections
        - Position-specific AA features (epitope dim: 820)
        - Protein conservation features (protein dim: 324)
        - AdamW optimizer

Run from project root:
    uv run python scripts/07_improve_v3.py
"""

import sys
import re
import json
import time
from pathlib import Path
from collections import defaultdict

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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# ── Setup ─────────────────────────────────────────────────────────────────────

console = Console()
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EMBED_DIR     = PROCESSED_DIR / "embeddings"
GRAPH_DIR     = PROCESSED_DIR / "graph"
MODELS_DIR    = PROJECT_ROOT / "outputs" / "models"
FIGURES_DIR   = PROJECT_ROOT / "outputs" / "figures"
OUT_DIR       = PROJECT_ROOT / "outputs" / "vaccine_candidates"

for d in [GRAPH_DIR, MODELS_DIR, FIGURES_DIR, OUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
logger.add(PROJECT_ROOT / "outputs" / "phase7_v3.log", rotation="5 MB")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.family": "DejaVu Sans",
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "figure.facecolor": "white",
})

# ── Hyperparameters ───────────────────────────────────────────────────────────

HP = {
    "hidden_dim":    256,
    "num_heads":     4,
    "num_layers":    4,
    "dropout":       0.3,
    "lr":            1e-3,      # back to original lr — uniform for all layers
    "weight_decay":  1e-4,
    "epochs":        200,
    "patience":      25,
    # FIX 1: alpha=0.80 → minority (positive) class gets 80% weight
    # With 9.6:1 imbalance, positives need strong upweighting
    "focal_alpha":   0.80,
    "focal_gamma":   2.0,
    "sim_threshold": 0.70,
    "knn_k":         8,
    "random_seed":   42,
}

# ── Constants ─────────────────────────────────────────────────────────────────

AA_ORDER    = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX   = {aa: i for i, aa in enumerate(AA_ORDER)}
N_AA        = 20
MAX_EPI_LEN = 25

TB_ESSENTIAL_GENES = {
    "kasA","kasB","accD6","acpM","fabD","plsB","murA","murB","murC","murD",
    "murE","murF","murG","murI","ftsZ","ddlA","dnaA","dnaN","dnaB","dnaE1",
    "dnaE2","gyrA","gyrB","ligA","ssb","dnaG","rpoB","rpoC","rpsA","rpsB",
    "rplB","rplC","rplD","rplE","rplF","rplL","tufA","fusA","infB","truB",
    "fba","tpi","gap","pgk","eno","pykA","gnd","icl1","icl2","sucA","sucB",
    "sucD","sdhA","atpA","atpB","atpC","atpD","atpE","atpF","atpG","atpH",
    "nuoA","nuoB","nuoD","qcrB","ctaD","trpA","trpB","trpC","trpD","trpE",
    "hisA","hisB","hisC","hisD","hisE","aroA","aroB","aroC","aroD","aroE",
    "inhA","fabG1","hadA","hadB","hadC","katG","embA","embB","embC","pncA",
    "eccB1","eccCa1","eccCb1","eccD1","eccE1","esxA","esxB","esxH","esxU",
    "esxV","esxW",
}

HLA_SUPERTYPES = {
    "A01": ["A*01:01","A*01:02","A*01:03","A*36:01"],
    "A02": ["A*02:01","A*02:02","A*02:03","A*02:04","A*02:05",
            "A*02:06","A*02:07","A*02:11","A*02:12","A*69:01"],
    "A03": ["A*03:01","A*11:01","A*31:01","A*33:01","A*68:01","A*74:01"],
    "A24": ["A*24:02","A*24:03","A*24:07","A*23:01"],
    "B07": ["B*07:02","B*35:01","B*51:01","B*53:01","B*54:01","B*55:01"],
    "B08": ["B*08:01","B*14:02","B*38:01","B*39:01","B*40:01","B*40:02"],
    "B27": ["B*27:02","B*27:04","B*27:05","B*27:07"],
    "B44": ["B*37:01","B*44:02","B*44:03","B*45:01"],
    "B62": ["B*15:01","B*15:02","B*15:03","B*46:01","B*52:01"],
}

# ── Utility ───────────────────────────────────────────────────────────────────

def assign_mhc_class(length: int) -> str:
    return "Class I (CD8+)" if length <= 11 else "Class II (CD4+)"


def position_aa_features(seq: str) -> np.ndarray:
    feat = np.zeros((MAX_EPI_LEN, N_AA), dtype=np.float32)
    for i, aa in enumerate(seq.upper()[:MAX_EPI_LEN]):
        if aa in AA_TO_IDX:
            feat[i, AA_TO_IDX[aa]] = 1.0
    return feat.flatten()


def build_position_features(seqs: list) -> np.ndarray:
    logger.info(f"  Building position-AA features for {len(seqs):,} epitopes...")
    return np.array([position_aa_features(s) for s in seqs], dtype=np.float32)


def build_conservation_features(meta_prot: pd.DataFrame) -> np.ndarray:
    drug_tgts = {"inhA","katG","rpoB","rpoC","embA","embB","embC","pncA","gyrA","gyrB"}
    esx_prots = {"esxA","esxB","esxH","esxT","esxU","esxV","esxW","mpt70","mpt83"}
    max_len   = meta_prot["seq_length"].max()
    feats = np.zeros((len(meta_prot), 4), dtype=np.float32)
    for i, row in meta_prot.iterrows():
        gene = str(row.get("gene_name","")).lower().strip()
        feats[i, 0] = 1.0 if gene in TB_ESSENTIAL_GENES else 0.0
        feats[i, 1] = 1.0 if gene in esx_prots else 0.0
        feats[i, 2] = 1.0 if gene in drug_tgts else 0.0
        feats[i, 3] = row["seq_length"] / max_len
    logger.info(f"  Essential proteins: {(feats[:,0]==1).sum():,} | "
                f"Drug targets: {(feats[:,2]==1).sum():,}")
    return feats

# ── Improvement 1: Focal Loss (correctly configured) ─────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.

    alpha=0.80 means:
        positive class (immunogenic) gets weight 0.80
        negative class gets weight 0.20
    This strongly prioritizes learning from the minority positive class.

    gamma=2.0: standard focusing parameter — suppresses easy negatives,
    amplifies hard misclassified examples.
    """
    def __init__(self, alpha: float = 0.80, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs   = torch.sigmoid(logits)
        bce     = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t     = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss    = alpha_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()

# ── Load data ─────────────────────────────────────────────────────────────────

def load_data() -> dict:
    logger.info("Loading embeddings with v3 enhancements...")

    emb_pos  = np.load(str(EMBED_DIR / "epitopes_positive.npy"))
    emb_neg  = np.load(str(EMBED_DIR / "epitopes_negative.npy"))
    meta_pos = pd.read_csv(EMBED_DIR / "epitopes_positive_meta.csv")
    meta_neg = pd.read_csv(EMBED_DIR / "epitopes_negative_meta.csv")
    emb_epi  = np.vstack([emb_pos, emb_neg])
    meta_epi = pd.concat([meta_pos, meta_neg], ignore_index=True)
    meta_epi["global_idx"] = range(len(meta_epi))

    pos_feats    = build_position_features(meta_epi["epitope_seq"].tolist())
    emb_epi_full = np.hstack([emb_epi, pos_feats])
    logger.info(f"  Epitope dim: {emb_epi.shape[1]} ESM + {pos_feats.shape[1]} pos = {emb_epi_full.shape[1]}")

    emb_prot  = np.load(str(EMBED_DIR / "tb_proteins.npy"))
    meta_prot = pd.read_csv(EMBED_DIR / "tb_proteins_meta.csv")
    cons_feats    = build_conservation_features(meta_prot)
    emb_prot_full = np.hstack([emb_prot, cons_feats])
    logger.info(f"  Protein dim: {emb_prot.shape[1]} ESM + {cons_feats.shape[1]} cons = {emb_prot_full.shape[1]}")

    emb_hla  = np.load(str(EMBED_DIR / "hla_sample.npy"))
    meta_hla = pd.read_csv(EMBED_DIR / "hla_sample_meta.csv")
    logger.info(f"  HLA dim: {emb_hla.shape[1]}")

    df_vjdb     = pd.read_csv(PROCESSED_DIR / "vjdb_tb_human_clean.tsv", sep="\t")
    unique_cdr3 = df_vjdb["cdr3"].dropna().unique()

    def cdr3_feat(seq):
        seq = str(seq).upper()
        c = np.array([seq.count(aa) for aa in AA_ORDER], dtype=np.float32)
        c /= max(len(seq), 1)
        return np.concatenate([c, [len(seq) / 30.0]])

    cdr3_feats = np.array([cdr3_feat(s) for s in unique_cdr3])
    meta_tcr   = pd.DataFrame({"cdr3": unique_cdr3, "embed_idx": range(len(unique_cdr3))})
    logger.info(f"  TCR dim: {cdr3_feats.shape[1]}")
    logger.info(f"  Nodes — Epitopes: {len(meta_epi):,} | Proteins: {len(meta_prot):,} | "
                f"HLA: {len(meta_hla):,} | TCR: {len(meta_tcr):,}")

    return {
        "epitope": {"embeddings": emb_epi_full, "esm_emb": emb_epi,
                    "meta": meta_epi, "n": len(meta_epi)},
        "protein": {"embeddings": emb_prot_full, "meta": meta_prot, "n": len(meta_prot)},
        "hla":     {"embeddings": emb_hla,        "meta": meta_hla, "n": len(meta_hla)},
        "tcr":     {"embeddings": cdr3_feats,     "meta": meta_tcr, "n": len(meta_tcr),
                    "df_vjdb": df_vjdb},
    }

# ── Edge builders ─────────────────────────────────────────────────────────────

def sim_edges(esm_emb: np.ndarray, k: int, thresh: float) -> torch.Tensor:
    norm = esm_emb.astype(np.float32)
    norm = norm / (np.linalg.norm(norm, axis=1, keepdims=True) + 1e-8)
    src, dst = [], []
    for i in range(0, len(norm), 1000):
        b    = norm[i:i+1000]
        sims = b @ norm.T
        np.fill_diagonal(sims[:, i:i+len(b)], 0)
        top  = np.argsort(sims, axis=1)[:, -k:]
        for li in range(len(b)):
            gi = i + li
            for ni in top[li]:
                if sims[li, ni] >= thresh and ni != gi:
                    src.append(gi); dst.append(int(ni))
    logger.info(f"  Similarity edges: {len(src):,}")
    return torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)


def hla_supertype_edges(meta_hla: pd.DataFrame) -> torch.Tensor:
    node_to_st = {}
    for i, row in meta_hla.iterrows():
        m = re.search(r"([A-Z]\*\d+:\d+)", str(row.get("allele", "")))
        if m:
            allele = m.group(1)
            for st, members in HLA_SUPERTYPES.items():
                if any(allele.startswith(mb[:6]) for mb in members):
                    node_to_st[i] = st; break
    groups = defaultdict(list)
    for node, st in node_to_st.items():
        groups[st].append(node)
    src, dst = [], []
    for nodes in groups.values():
        for i in range(len(nodes)):
            for j in range(i+1, min(i+10, len(nodes))):
                src += [nodes[i], nodes[j]]; dst += [nodes[j], nodes[i]]
    logger.info(f"  HLA supertype edges: {len(src):,}")
    return torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)


def protein_epitope_edges(data: dict) -> torch.Tensor:
    meta_epi  = data["epitope"]["meta"]
    meta_prot = data["protein"]["meta"]
    iedb_all  = pd.concat([
        pd.read_csv(PROCESSED_DIR / "iedb_positive_clean.csv"),
        pd.read_csv(PROCESSED_DIR / "iedb_negative_clean.csv"),
    ], ignore_index=True)
    mol_col = next((c for c in iedb_all.columns if "source_molecule" in c), None)
    if mol_col is None:
        return torch.zeros((2, 0), dtype=torch.long)
    seq_to_src = dict(zip(iedb_all["epitope_seq"], iedb_all[mol_col].fillna("")))
    pidx = {}
    for i, row in meta_prot.iterrows():
        if row["gene_name"]: pidx[row["gene_name"].upper()] = i
        pidx[str(row["protein_name"]).split()[0].upper().rstrip(",")] = i
    src, dst = [], []
    for ei, row in meta_epi.iterrows():
        source = str(seq_to_src.get(row["epitope_seq"], "")).upper()
        for word in source.split():
            if word.rstrip(",.") in pidx:
                src.append(pidx[word.rstrip(",.")]);  dst.append(ei); break
    return torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)


def epitope_tcr_edges(data: dict) -> torch.Tensor:
    meta_epi = data["epitope"]["meta"]
    meta_tcr = data["tcr"]["meta"]
    df_vjdb  = data["tcr"]["df_vjdb"]
    epi_idx  = dict(zip(meta_epi["epitope_seq"].str.upper(), meta_epi["global_idx"]))
    tcr_idx  = dict(zip(meta_tcr["cdr3"].str.upper(), meta_tcr["embed_idx"]))
    src, dst = [], []
    for _, row in df_vjdb.iterrows():
        ei = epi_idx.get(str(row.get("epitope","")).upper().strip())
        ti = tcr_idx.get(str(row.get("cdr3","")).upper().strip())
        if ei is not None and ti is not None:
            src.append(ei); dst.append(ti)
    return torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)


def epitope_hla_edges(data: dict) -> torch.Tensor:
    en = data["epitope"]["esm_emb"].astype(np.float32)
    hn = data["hla"]["embeddings"].astype(np.float32)
    en = en / (np.linalg.norm(en, axis=1, keepdims=True) + 1e-8)
    hn = hn / (np.linalg.norm(hn, axis=1, keepdims=True) + 1e-8)
    src, dst = [], []
    for i in range(0, len(en), 500):
        b    = en[i:i+500]
        sims = b @ hn.T
        top3 = np.argsort(sims, axis=1)[:, -3:]
        for li in range(len(b)):
            for hi in top3[li]:
                if sims[li, hi] > 0.5:
                    src.append(i+li); dst.append(int(hi))
    return torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)


def build_graph(data: dict) -> HeteroData:
    console.rule("[yellow]Building graph v3[/yellow]")
    graph = HeteroData()

    graph["epitope"].x   = torch.tensor(data["epitope"]["embeddings"], dtype=torch.float32)
    graph["epitope"].y   = torch.tensor(data["epitope"]["meta"]["label"].values, dtype=torch.long)
    graph["epitope"].seq = data["epitope"]["meta"]["epitope_seq"].tolist()
    graph["epitope"].mhc = [assign_mhc_class(len(s)) for s in data["epitope"]["meta"]["epitope_seq"]]
    graph["protein"].x         = torch.tensor(data["protein"]["embeddings"], dtype=torch.float32)
    graph["protein"].gene_name = data["protein"]["meta"]["gene_name"].tolist()
    graph["hla"].x      = torch.tensor(data["hla"]["embeddings"], dtype=torch.float32)
    graph["hla"].allele = data["hla"]["meta"]["allele"].tolist()
    graph["tcr"].x    = torch.tensor(data["tcr"]["embeddings"], dtype=torch.float32)
    graph["tcr"].cdr3 = data["tcr"]["meta"]["cdr3"].tolist()

    for nt in graph.node_types:
        logger.info(f"  {nt} feature dim: {graph[nt].x.shape[1]}")

    logger.info("Building edges...")
    ei_pe = protein_epitope_edges(data)
    graph["protein", "source_of",    "epitope"].edge_index = ei_pe
    logger.info(f"  protein→epitope: {ei_pe.shape[1]:,}")
    ei_eh = epitope_hla_edges(data)
    graph["epitope", "binds_to",     "hla"].edge_index = ei_eh
    logger.info(f"  epitope→hla:    {ei_eh.shape[1]:,}")
    ei_et = epitope_tcr_edges(data)
    graph["epitope", "recognized_by","tcr"].edge_index = ei_et
    logger.info(f"  epitope→tcr:    {ei_et.shape[1]:,}")
    ei_sim = sim_edges(data["epitope"]["esm_emb"], HP["knn_k"], HP["sim_threshold"])
    graph["epitope", "similar_to",   "epitope"].edge_index = ei_sim
    ei_st = hla_supertype_edges(data["hla"]["meta"])
    graph["hla", "same_supertype",   "hla"].edge_index = ei_st

    total = sum(graph[et].edge_index.shape[1] for et in graph.edge_types)
    logger.info(f"  Total edges: {total:,}")
    torch.save(graph, str(GRAPH_DIR / "heterogeneous_graph_v3.pt"))
    return graph

# ── Model ─────────────────────────────────────────────────────────────────────

class EpitopeGNN_v3(nn.Module):
    """
    Wider (256) + Deeper (4 layers) HAN with per-node-type input projections.
    Each node type projected from its own feature dim to hidden_dim before HANConv.
    """
    def __init__(self, node_in_dims: dict, hidden_dim: int, conv_out_dim: int,
                 num_heads: int, num_layers: int, dropout: float, metadata: tuple):
        super().__init__()
        self.dropout = dropout
        node_types   = metadata[0]

        # Per-node-type projection — each maps its own dim → hidden_dim
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
                HANConv(in_ch, conv_out_dim, heads=num_heads,
                        dropout=dropout, metadata=metadata)
            )
            self.norms.append(nn.ModuleDict({
                nt: nn.LayerNorm(conv_out_dim) for nt in node_types
            }))
            self.projs.append(
                nn.ModuleDict({
                    nt: nn.Linear(in_ch, conv_out_dim, bias=False) for nt in node_types
                }) if conv_out_dim != in_ch else None
            )

        self.classifier = nn.Sequential(
            nn.Linear(conv_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1),
        )

    def forward(self, x_dict, edge_index_dict):
        h = {nt: proj(x_dict[nt]) for nt, proj in self.input_proj.items() if nt in x_dict}

        for i, conv in enumerate(self.convs):
            h_new = conv(h, edge_index_dict)
            for nt in h_new:
                if h_new[nt] is None: continue
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
                h_new[nt] = F.relu(F.dropout(h_new[nt], p=self.dropout, training=self.training))
            for nt in h_new:
                if h_new[nt] is not None: h[nt] = h_new[nt]

        return self.classifier(h["epitope"]).squeeze(-1)


def probe_conv_dim(metadata, hidden_dim, num_heads):
    dummy_x  = {nt: torch.zeros(2, hidden_dim) for nt in metadata[0]}
    dummy_ei = {et: torch.zeros(2, 0, dtype=torch.long) for et in metadata[1]}
    try:
        conv = HANConv(hidden_dim, hidden_dim, heads=num_heads, metadata=metadata)
        out  = conv(dummy_x, dummy_ei)
        for nt in metadata[0]:
            if nt in out and out[nt] is not None: return out[nt].shape[1]
    except Exception: pass
    return hidden_dim

# ── Training ──────────────────────────────────────────────────────────────────

def make_splits(graph):
    labels = graph["epitope"].y.cpu().numpy(); n = len(labels)
    idx    = np.arange(n)
    ti, tmp = train_test_split(idx, test_size=0.30, stratify=labels, random_state=42)
    vi, xi  = train_test_split(tmp, test_size=0.50, stratify=labels[tmp], random_state=42)
    tm = torch.zeros(n, dtype=torch.bool); tm[ti] = True
    vm = torch.zeros(n, dtype=torch.bool); vm[vi] = True
    xm = torch.zeros(n, dtype=torch.bool); xm[xi] = True
    logger.info(f"  Train {tm.sum():,} | Val {vm.sum():,} | Test {xm.sum():,}")
    logger.info(f"  Positives in train: {labels[ti].sum():,}")
    return tm.to(device), vm.to(device), xm.to(device)


def train_one_epoch(model, graph, mask, optimizer, criterion):
    model.train(); optimizer.zero_grad()
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
    console.rule("[yellow]Training v3[/yellow]")

    criterion = FocalLoss(alpha=HP["focal_alpha"], gamma=HP["focal_gamma"])

    # FIX 3: uniform LR for all parameters — no layer-wise LR
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"]
    )

    # FIX 2: ReduceLROnPlateau — only reduces when val AUROC stalls
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
        task = prog.add_task("v3", total=HP["epochs"],
                             ep=0, ep_t=HP["epochs"], auroc=0.0, tloss=0.0)

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
                logger.info(f"  Ep {epoch:3d} | loss={tl:.4f} | "
                            f"auroc={vm['auroc']:.4f} | auprc={vm['auprc']:.4f} | "
                            f"f1={vm['f1']:.4f}")

            if patience_count >= HP["patience"]:
                logger.info(f"  Early stop epoch {epoch}, best AUROC={best_auroc:.4f}")
                break

    model.load_state_dict(best_state)
    logger.info(f"  Best: epoch {best_epoch}, val AUROC={best_auroc:.4f}")

    torch.save(
        {"model_state": best_state, "hyperparams": HP,
         "best_epoch": best_epoch, "best_val_auroc": best_auroc,
         "node_in_dims": {nt: graph[nt].x.shape[1] for nt in graph.node_types}},
        str(MODELS_DIR / "best_model_v3.pt"),
    )
    with open(MODELS_DIR / "training_history_v3.json", "w") as f:
        json.dump({k: [float(v) for v in vs] for k, vs in history.items()}, f, indent=2)

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
    meta_prot = pd.read_csv(EMBED_DIR / "tb_proteins_meta.csv")
    vjdb_eps  = set(data["tcr"]["df_vjdb"]["epitope"].astype(str).str.upper().str.strip())

    prot_ei     = graph["protein", "source_of", "epitope"].edge_index
    epi_to_prot = {}
    if prot_ei.shape[1] > 0:
        for i in range(prot_ei.shape[1]):
            ei = int(prot_ei[1, i])
            if ei not in epi_to_prot: epi_to_prot[ei] = int(prot_ei[0, i])

    hla_ei    = graph["epitope", "binds_to", "hla"].edge_index
    hla_count = np.zeros(len(seqs), dtype=int)
    if hla_ei.shape[1] > 0:
        for i in range(hla_ei.shape[1]):
            ei = int(hla_ei[0, i])
            if ei < len(hla_count): hla_count[ei] += 1
    max_hla = max(hla_count.max(), 1)

    rows = []
    for i, (seq, score, label, mhc) in enumerate(zip(seqs, scores, labels, mhcs)):
        tcr_ev  = 1 if seq.upper() in vjdb_eps else 0
        hla_cov = hla_count[i] / max_hla
        gene = prot_name = ""
        if i in epi_to_prot:
            pi = epi_to_prot[i]
            if pi < len(meta_prot):
                gene      = str(meta_prot.iloc[pi].get("gene_name", ""))
                prot_name = str(meta_prot.iloc[pi].get("protein_name", ""))[:50]
        is_ess   = int(gene.lower() in TB_ESSENTIAL_GENES)
        composite = 0.50*float(score) + 0.25*tcr_ev + 0.15*hla_cov + 0.10*is_ess
        rows.append({
            "epitope_seq": seq, "seq_length": len(seq), "mhc_class": mhc,
            "gnn_score": float(score), "tcr_evidence": tcr_ev,
            "hla_coverage_score": float(hla_cov), "hla_neighbors": int(hla_count[i]),
            "essential_gene": is_ess, "composite_score": float(composite),
            "true_label": int(label), "source_gene": gene, "source_protein": prot_name,
        })

    df = pd.DataFrame(rows)
    before = len(df)
    df     = df.sort_values("composite_score", ascending=False)
    df     = df.drop_duplicates(subset=["epitope_seq"], keep="first")
    logger.info(f"  Dedup: {before:,} → {len(df):,} unique sequences")

    df_cand = df[df["gnn_score"] > 0.5].copy().reset_index(drop=True)
    df_cand["rank"] = range(1, len(df_cand) + 1)

    c1 = (df_cand["mhc_class"] == "Class I (CD8+)").sum()
    c2 = (df_cand["mhc_class"] == "Class II (CD4+)").sum()
    logger.info(f"  Candidates: {len(df_cand):,} | Class I: {c1:,} | Class II: {c2:,}")
    logger.info(f"  TCR-confirmed: {df_cand['tcr_evidence'].sum():,}")
    logger.info(f"  Essential gene: {df_cand['essential_gene'].sum():,}")
    logger.info(f"  True pos in top 50: {df_cand.head(50)['true_label'].sum():,} / 50")
    return df_cand

# ── Plots ─────────────────────────────────────────────────────────────────────

def save_plots(history, best_epoch, test_m):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("GNN v3: Focal Loss (α=0.80) + 256dim + 4 Layers", fontweight="bold")
    ep = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(ep, history["train_loss"], "#2E86AB", label="Train", linewidth=1.5)
    axes[0].plot(ep, history["val_loss"],   "#E84855", label="Val",   linewidth=1.5)
    axes[0].axvline(best_epoch, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_title("Focal Loss (α=0.80)"); axes[0].legend(fontsize=9)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Focal Loss")

    axes[1].plot(ep, history["val_auroc"], "#3BB273", linewidth=1.5)
    axes[1].axhline(max(history["val_auroc"]), color="#3BB273", linestyle=":",
                    label=f"Best={max(history['val_auroc']):.4f}")
    axes[1].axvline(best_epoch, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_title("Val AUROC"); axes[1].set_ylim(0, 1); axes[1].legend(fontsize=9)

    axes[2].plot(ep, history["val_auprc"], "#7B4F9E", linewidth=1.5)
    axes[2].axhline(max(history["val_auprc"]), color="#7B4F9E", linestyle=":",
                    label=f"Best={max(history['val_auprc']):.4f}")
    axes[2].axvline(best_epoch, color="gray", linestyle="--", linewidth=0.8)
    axes[2].set_title("Val AUPRC"); axes[2].set_ylim(0, 1); axes[2].legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "16_v3_training.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved 16_v3_training.png")

    from sklearn.metrics import roc_curve, precision_recall_curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Test Set — GNN v3", fontweight="bold")
    fpr, tpr, _ = roc_curve(test_m["labels"], test_m["probs"])
    axes[0].plot(fpr, tpr, "#3BB273", linewidth=2, label=f"AUROC={test_m['auroc']:.4f}")
    axes[0].plot([0,1],[0,1], "gray", linestyle="--", linewidth=0.8, label="Random")
    axes[0].fill_between(fpr, tpr, alpha=0.1, color="#3BB273")
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC Curve"); axes[0].legend()
    p, r, _ = precision_recall_curve(test_m["labels"], test_m["probs"])
    axes[1].plot(r, p, "#7B4F9E", linewidth=2, label=f"AUPRC={test_m['auprc']:.4f}")
    axes[1].axhline(test_m["labels"].mean(), color="gray", linestyle="--",
                    label=f"Random={test_m['labels'].mean():.3f}")
    axes[1].fill_between(r, p, alpha=0.1, color="#7B4F9E")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall"); axes[1].legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "17_v3_roc_pr.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved 17_v3_roc_pr.png")

# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(test_m, df_cand):
    console.rule("[bold green]GNN v3 — Final Results[/bold green]")

    t = Table(title="v1 vs v2 vs v3", header_style="bold cyan", show_lines=True)
    t.add_column("Metric",   style="white",      min_width=18)
    t.add_column("v1",       style="yellow",     min_width=10)
    t.add_column("v2",       style="yellow",     min_width=10)
    t.add_column("v3",       style="bold green", min_width=10)
    t.add_column("Winner",   style="dim",        min_width=8)

    v1 = {"auroc":0.8928,"auprc":0.5259,"f1":0.4601,"rec":0.8309}
    v2 = {"auroc":0.8811,"auprc":0.5085,"f1":0.4258,"rec":0.8338}

    for label, key in [("AUROC","auroc"),("AUPRC","auprc"),("F1","f1"),("Recall","rec")]:
        vals = [v1[key], v2[key], test_m[key]]
        best = ["v1","v2","v3"][int(np.argmax(vals))]
        t.add_row(label, f"{v1[key]:.4f}", f"{v2[key]:.4f}", f"{test_m[key]:.4f}", best)
    console.print(t)

    c1 = df_cand[df_cand["mhc_class"] == "Class I (CD8+)"]
    c2 = df_cand[df_cand["mhc_class"] == "Class II (CD4+)"]

    console.print(f"\n[bold]v3 candidate summary:[/bold]")
    console.print(f"  Total unique candidates:   {len(df_cand):,}")
    console.print(f"  Class I  (CD8+):           {len(c1):,}")
    console.print(f"  Class II (CD4+):           {len(c2):,}")
    console.print(f"  TCR-confirmed:             {df_cand['tcr_evidence'].sum():,}")
    console.print(f"  Essential gene candidates: {df_cand['essential_gene'].sum():,}")
    console.print(f"  True pos in top 50:        {df_cand.head(50)['true_label'].sum():,} / 50")

    console.print("\n[bold]Top 5 Class I (CD8+):[/bold]")
    for _, row in c1.head(5).iterrows():
        tags = (" [TCR]" if row["tcr_evidence"] else "") + (" [essential]" if row["essential_gene"] else "")
        console.print(f"  #{int(row['rank']):3d} {row['epitope_seq']:<22} "
                      f"score={row['composite_score']:.4f}  gene={row['source_gene'] or '?'}{tags}")

    console.print("\n[bold]Top 5 Class II (CD4+):[/bold]")
    for _, row in c2.head(5).iterrows():
        tags = (" [TCR]" if row["tcr_evidence"] else "") + (" [essential]" if row["essential_gene"] else "")
        console.print(f"  #{int(row['rank']):3d} {row['epitope_seq']:<25} "
                      f"score={row['composite_score']:.4f}  gene={row['source_gene'] or '?'}{tags}")

    console.print(f"\n[bold cyan]Next:[/bold cyan] uv run python scripts/08_multi_epitope_assembly.py\n")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    console.rule("[bold cyan]Phase 7 v3 — Fixed: Focal Loss α=0.80, ReduceLROnPlateau, Uniform LR[/bold cyan]")
    console.print(
        "\n[bold]3 fixes applied:[/bold]\n"
        "  FIX 1: focal_alpha 0.25 → 0.80 (positives now get 80% weight)\n"
        "  FIX 2: CosineAnnealing → ReduceLROnPlateau\n"
        "  FIX 3: Layer-wise LR removed — uniform lr=1e-3 for all layers\n\n"
        "[bold]Kept from v3:[/bold]\n"
        "  hidden_dim=256, num_layers=4, AdamW, position-AA features, conservation features\n"
    )

    t0   = time.time()
    data = load_data()
    graph = build_graph(data).to(device)

    node_in_dims = {nt: graph[nt].x.shape[1] for nt in graph.node_types}
    logger.info(f"  Node dims: {node_in_dims}")

    console.rule("[yellow]Building model v3[/yellow]")
    conv_out = probe_conv_dim(graph.metadata(), HP["hidden_dim"], HP["num_heads"])
    model = EpitopeGNN_v3(
        node_in_dims=node_in_dims, hidden_dim=HP["hidden_dim"],
        conv_out_dim=conv_out, num_heads=HP["num_heads"],
        num_layers=HP["num_layers"], dropout=HP["dropout"],
        metadata=graph.metadata(),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {n_params:,} | Conv out: {conv_out}")

    train_mask, val_mask, test_mask = make_splits(graph)
    history, best_epoch = run_training(model, graph, train_mask, val_mask)

    criterion    = FocalLoss(HP["focal_alpha"], HP["focal_gamma"])
    val_metrics  = evaluate(model, graph, val_mask,  criterion)
    test_metrics = evaluate(model, graph, test_mask, criterion)

    logger.info(f"  Test AUROC:  {test_metrics['auroc']:.4f}")
    logger.info(f"  Test AUPRC:  {test_metrics['auprc']:.4f}")
    logger.info(f"  Test F1:     {test_metrics['f1']:.4f}")
    logger.info(f"  Test Recall: {test_metrics['rec']:.4f}")

    scores  = score_all(model, graph)
    df_cand = rank_candidates(graph, scores, data)

    df_cand.to_csv(OUT_DIR / "top_candidates_v3.csv", index=False)
    df_cand[df_cand["mhc_class"]=="Class I (CD8+)"].head(25).to_csv(
        OUT_DIR / "top25_classI_v3.csv", index=False)
    df_cand[df_cand["mhc_class"]=="Class II (CD4+)"].head(25).to_csv(
        OUT_DIR / "top25_classII_v3.csv", index=False)
    logger.info("  Saved v3 CSVs")

    save_plots(history, best_epoch, test_metrics)
    print_summary(test_metrics, df_cand)
    logger.info(f"  Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()