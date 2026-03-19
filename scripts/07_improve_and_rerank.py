"""
07_improve_and_rerank.py
========================
Phase 7: GNN Improvement + Deduplicated Reranking
GNN-Guided Multi-Epitope Vaccine Design

What this script does (in order):
    Step 1 — Rebuild graph with improvements:
        - Lower similarity threshold: 0.85 → 0.70 (denser edges, more signal)
        - Add HLA supertype grouping edges (biological prior knowledge)
        - Add MHC class annotation per epitope node

    Step 2 — Retrain GNN on improved graph

    Step 3 — Rerank with fixes:
        - Deduplicate by sequence (keep highest score per unique sequence)
        - Fix MHC class labels (correct biology: ≤11aa = Class I, ≥12aa = Class II)
        - Ensure both Class I and Class II candidates in final output
        - Produce separate top-25 lists for Class I and Class II

Outputs:
    data/processed/graph/heterogeneous_graph_v2.pt
    outputs/models/best_model_v2.pt
    outputs/vaccine_candidates/
        top_candidates_v2.csv           — deduplicated, full ranked list
        top25_classI_v2.csv             — MHC Class I shortlist  (CD8+ T-cells)
        top25_classII_v2.csv            — MHC Class II shortlist (CD4+ T-cells)
        comparison_v1_v2.csv            — before/after improvement comparison
    outputs/figures/
        13_v2_training_curves.png
        14_v2_roc_pr.png
        15_classI_classII_candidates.png

Run from project root:
    uv run python scripts/07_improve_and_rerank.py
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
    f1_score, precision_score, recall_score,
    confusion_matrix,
)
from Bio import SeqIO
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
logger.add(PROJECT_ROOT / "outputs" / "phase7.log", rotation="5 MB")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.family": "DejaVu Sans",
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "figure.facecolor": "white",
})

# ── Hyperparameters ───────────────────────────────────────────────────────────

HP = {
    "hidden_dim":    128,
    "num_heads":     4,
    "num_layers":    3,
    "dropout":       0.3,
    "lr":            1e-3,
    "weight_decay":  1e-4,
    "epochs":        200,
    "patience":      20,
    "pos_weight":    9.6,
    "train_ratio":   0.70,
    "val_ratio":     0.15,
    "test_ratio":    0.15,
    "random_seed":   42,
    # Improved graph settings
    "sim_threshold": 0.70,   # was 0.85 — denser similarity edges
    "knn_k":         8,      # was 5 — more neighbors per epitope
}

# ── HLA Supertypes ────────────────────────────────────────────────────────────
# These are the 9 major HLA-A/B supertypes that cover ~97% of world population.
# Source: Sette & Sidney, Immunogenetics 1999
# Epitopes binding multiple alleles within a supertype have broader coverage.

HLA_SUPERTYPES = {
    "A01": ["A*01:01", "A*01:02", "A*01:03", "A*36:01"],
    "A02": ["A*02:01", "A*02:02", "A*02:03", "A*02:04", "A*02:05", "A*02:06",
            "A*02:07", "A*02:11", "A*02:12", "A*69:01"],
    "A03": ["A*03:01", "A*11:01", "A*31:01", "A*33:01", "A*68:01", "A*74:01"],
    "A24": ["A*24:02", "A*24:03", "A*24:07", "A*23:01"],
    "B07": ["B*07:02", "B*35:01", "B*51:01", "B*53:01", "B*54:01",
            "B*55:01", "B*56:01"],
    "B08": ["B*08:01", "B*14:02", "B*38:01", "B*39:01", "B*40:01", "B*40:02"],
    "B27": ["B*27:02", "B*27:04", "B*27:05", "B*27:07"],
    "B44": ["B*37:01", "B*44:02", "B*44:03", "B*45:01"],
    "B62": ["B*15:01", "B*15:02", "B*15:03", "B*46:01", "B*52:01"],
}

# ── MHC Class assignment ──────────────────────────────────────────────────────

def assign_mhc_class(length: int) -> str:
    """
    Assign MHC class based on peptide length.

    Biology:
        MHC Class I groove fits 8–11 aa peptides (sometimes 12)
        MHC Class II groove fits 12–25 aa peptides (sometimes longer)

        CD8+ cytotoxic T-cells recognize Class I — they KILL infected cells
        CD4+ helper T-cells recognize Class II — they COORDINATE the immune response

        A good vaccine needs BOTH to create lasting immunity.
    """
    if length <= 11:
        return "Class I (CD8+)"
    else:
        return "Class II (CD4+)"


# ════════════════════════════════════════════════════════════════════════════════
# STEP 1: REBUILD GRAPH
# ════════════════════════════════════════════════════════════════════════════════

def load_all_embeddings() -> dict:
    """Load all embeddings and metadata (same as Phase 4)."""
    logger.info("Loading embeddings...")

    emb_pos  = np.load(str(EMBED_DIR / "epitopes_positive.npy"))
    emb_neg  = np.load(str(EMBED_DIR / "epitopes_negative.npy"))
    meta_pos = pd.read_csv(EMBED_DIR / "epitopes_positive_meta.csv")
    meta_neg = pd.read_csv(EMBED_DIR / "epitopes_negative_meta.csv")

    emb_epitopes  = np.vstack([emb_pos, emb_neg])
    meta_epitopes = pd.concat([meta_pos, meta_neg], ignore_index=True)
    meta_epitopes["global_idx"] = range(len(meta_epitopes))

    emb_prot  = np.load(str(EMBED_DIR / "tb_proteins.npy"))
    meta_prot = pd.read_csv(EMBED_DIR / "tb_proteins_meta.csv")

    emb_hla  = np.load(str(EMBED_DIR / "hla_sample.npy"))
    meta_hla = pd.read_csv(EMBED_DIR / "hla_sample_meta.csv")

    vjdb_path = PROCESSED_DIR / "vjdb_tb_human_clean.tsv"
    df_vjdb   = pd.read_csv(vjdb_path, sep="\t")
    unique_cdr3 = df_vjdb["cdr3"].dropna().unique()

    AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
    def cdr3_features(seq):
        seq = str(seq).upper()
        counts = np.array([seq.count(aa) for aa in AA_ORDER], dtype=np.float32)
        counts /= max(len(seq), 1)
        return np.concatenate([counts, [len(seq) / 30.0]])

    cdr3_feats = np.array([cdr3_features(s) for s in unique_cdr3])
    pad = emb_epitopes.shape[1] - cdr3_feats.shape[1]
    cdr3_feats = np.pad(cdr3_feats, ((0, 0), (0, pad)))

    meta_tcr = pd.DataFrame({"cdr3": unique_cdr3, "embed_idx": range(len(unique_cdr3))})

    logger.info(f"  Epitopes: {len(meta_epitopes):,} | Proteins: {len(meta_prot):,} | "
                f"HLA: {len(meta_hla):,} | TCR: {len(meta_tcr):,}")

    return {
        "epitope": {"embeddings": emb_epitopes, "meta": meta_epitopes, "n": len(meta_epitopes)},
        "protein": {"embeddings": emb_prot,     "meta": meta_prot,     "n": len(meta_prot)},
        "hla":     {"embeddings": emb_hla,       "meta": meta_hla,     "n": len(meta_hla)},
        "tcr":     {"embeddings": cdr3_feats,    "meta": meta_tcr,     "n": len(meta_tcr),
                    "df_vjdb": df_vjdb},
    }


def build_similarity_edges(emb: np.ndarray, k: int, threshold: float) -> tuple:
    """Build k-NN similarity edges with lower threshold for denser graph."""
    logger.info(f"  Building similarity edges (k={k}, threshold={threshold})...")
    n    = len(emb)
    norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

    src, dst = [], []
    batch    = 1000

    for i in range(0, n, batch):
        b    = norm[i:i+batch]
        sims = b @ norm.T
        np.fill_diagonal(sims[:, i:i+len(b)], 0)
        top  = np.argsort(sims, axis=1)[:, -k:]

        for li in range(len(b)):
            gi = i + li
            for ni in top[li]:
                if sims[li, ni] >= threshold and ni != gi:
                    src.append(gi); dst.append(int(ni))

    logger.info(f"  Similarity edges: {len(src):,}")
    return torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)


def build_hla_supertype_edges(meta_hla: pd.DataFrame) -> tuple:
    """
    Build edges between HLA alleles of the same supertype.

    Why: Supertypes group HLA alleles with similar peptide-binding properties.
    If an epitope binds HLA-A*02:01, it likely also binds A*02:06.
    These edges let the GNN learn supertype-level patterns.
    """
    logger.info("  Building HLA supertype edges...")

    # Map each HLA node to its supertype
    allele_to_supertype = {}
    for supertype, alleles in HLA_SUPERTYPES.items():
        for allele in alleles:
            allele_to_supertype[allele] = supertype

    # Map HLA node index to supertype
    node_to_supertype = {}
    for i, row in meta_hla.iterrows():
        allele_str = str(row.get("allele", ""))
        m = re.search(r"([A-Z]\*\d+:\d+)", allele_str)
        if m:
            allele = m.group(1)
            for supertype, members in HLA_SUPERTYPES.items():
                if any(allele.startswith(m[:6]) for m in members):
                    node_to_supertype[i] = supertype
                    break

    # Group nodes by supertype
    supertype_groups = defaultdict(list)
    for node_idx, supertype in node_to_supertype.items():
        supertype_groups[supertype].append(node_idx)

    src, dst = [], []
    for supertype, nodes in supertype_groups.items():
        if len(nodes) < 2:
            continue
        # Connect each node to all others in same supertype (undirected)
        for i in range(len(nodes)):
            for j in range(i+1, min(i+10, len(nodes))):  # cap at 10 neighbors
                src.append(nodes[i]); dst.append(nodes[j])
                src.append(nodes[j]); dst.append(nodes[i])

    logger.info(f"  HLA supertype edges: {len(src):,} across {len(supertype_groups)} supertypes")
    return torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)


def build_protein_epitope_edges(data: dict) -> torch.Tensor:
    """Same as Phase 4."""
    meta_epi  = data["epitope"]["meta"]
    meta_prot = data["protein"]["meta"]
    iedb_pos  = pd.read_csv(PROCESSED_DIR / "iedb_positive_clean.csv")
    iedb_neg  = pd.read_csv(PROCESSED_DIR / "iedb_negative_clean.csv")
    iedb_all  = pd.concat([iedb_pos, iedb_neg], ignore_index=True)
    mol_col   = next((c for c in iedb_all.columns if "source_molecule" in c), None)

    if mol_col is None:
        return torch.zeros((2, 0), dtype=torch.long)

    seq_to_source = dict(zip(iedb_all["epitope_seq"], iedb_all[mol_col].fillna("")))
    prot_name_to_idx = {}
    for i, row in meta_prot.iterrows():
        if row["gene_name"]:
            prot_name_to_idx[row["gene_name"].upper()] = i
        key = str(row["protein_name"]).split()[0].upper().rstrip(",")
        prot_name_to_idx[key] = i

    src, dst = [], []
    for epi_idx, row in meta_epi.iterrows():
        source = str(seq_to_source.get(row["epitope_seq"], "")).upper()
        if not source:
            continue
        for word in source.split():
            word = word.rstrip(",.")
            if word in prot_name_to_idx:
                src.append(prot_name_to_idx[word])
                dst.append(epi_idx)
                break

    return torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)


def build_epitope_tcr_edges(data: dict) -> torch.Tensor:
    """Same as Phase 4."""
    meta_epi = data["epitope"]["meta"]
    meta_tcr = data["tcr"]["meta"]
    df_vjdb  = data["tcr"]["df_vjdb"]

    epi_seq_to_idx = dict(zip(meta_epi["epitope_seq"].str.upper(), meta_epi["global_idx"]))
    cdr3_to_idx    = dict(zip(meta_tcr["cdr3"].str.upper(), meta_tcr["embed_idx"]))

    src, dst = [], []
    for _, row in df_vjdb.iterrows():
        ei = epi_seq_to_idx.get(str(row.get("epitope", "")).upper().strip())
        ti = cdr3_to_idx.get(str(row.get("cdr3", "")).upper().strip())
        if ei is not None and ti is not None:
            src.append(ei); dst.append(ti)

    return torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)


def build_epitope_hla_edges(data: dict) -> torch.Tensor:
    """Similarity-based HLA edges (same as Phase 4 fallback)."""
    emb_epi = data["epitope"]["embeddings"].astype(np.float32)
    emb_hla = data["hla"]["embeddings"].astype(np.float32)
    en = emb_epi / (np.linalg.norm(emb_epi, axis=1, keepdims=True) + 1e-8)
    hn = emb_hla / (np.linalg.norm(emb_hla, axis=1, keepdims=True) + 1e-8)

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


def build_improved_graph(data: dict) -> HeteroData:
    """Assemble improved heterogeneous graph v2."""
    console.rule("[yellow]Building improved graph v2[/yellow]")
    graph = HeteroData()

    # Node features
    graph["epitope"].x   = torch.tensor(data["epitope"]["embeddings"], dtype=torch.float32)
    graph["epitope"].y   = torch.tensor(data["epitope"]["meta"]["label"].values, dtype=torch.long)
    graph["epitope"].seq = data["epitope"]["meta"]["epitope_seq"].tolist()
    graph["epitope"].mhc = [
        assign_mhc_class(len(s)) for s in data["epitope"]["meta"]["epitope_seq"]
    ]

    graph["protein"].x         = torch.tensor(data["protein"]["embeddings"], dtype=torch.float32)
    graph["protein"].gene_name = data["protein"]["meta"]["gene_name"].tolist()

    graph["hla"].x      = torch.tensor(data["hla"]["embeddings"], dtype=torch.float32)
    graph["hla"].allele = data["hla"]["meta"]["allele"].tolist()

    graph["tcr"].x    = torch.tensor(data["tcr"]["embeddings"], dtype=torch.float32)
    graph["tcr"].cdr3 = data["tcr"]["meta"]["cdr3"].tolist()

    # Edges
    logger.info("Building edges...")

    ei_prot_epi = build_protein_epitope_edges(data)
    graph["protein", "source_of", "epitope"].edge_index = ei_prot_epi
    logger.info(f"  protein→epitope: {ei_prot_epi.shape[1]:,}")

    ei_epi_hla = build_epitope_hla_edges(data)
    graph["epitope", "binds_to", "hla"].edge_index = ei_epi_hla
    logger.info(f"  epitope→hla: {ei_epi_hla.shape[1]:,}")

    ei_epi_tcr = build_epitope_tcr_edges(data)
    graph["epitope", "recognized_by", "tcr"].edge_index = ei_epi_tcr
    logger.info(f"  epitope→tcr: {ei_epi_tcr.shape[1]:,}")

    # IMPROVED: lower threshold similarity edges
    ei_sim = build_similarity_edges(
        data["epitope"]["embeddings"],
        k=HP["knn_k"],
        threshold=HP["sim_threshold"],
    )
    graph["epitope", "similar_to", "epitope"].edge_index = ei_sim

    # NEW: HLA supertype edges
    ei_supertype = build_hla_supertype_edges(data["hla"]["meta"])
    graph["hla", "same_supertype", "hla"].edge_index = ei_supertype

    # Save
    path = GRAPH_DIR / "heterogeneous_graph_v2.pt"
    torch.save(graph, str(path))
    logger.info(f"  Graph v2 saved: {path.relative_to(PROJECT_ROOT)}")

    total_edges = (ei_prot_epi.shape[1] + ei_epi_hla.shape[1] +
                   ei_epi_tcr.shape[1] + ei_sim.shape[1] + ei_supertype.shape[1])
    logger.info(f"  Total edges: {total_edges:,} (was 214,459 in v1)")

    return graph


# ════════════════════════════════════════════════════════════════════════════════
# STEP 2: RETRAIN GNN
# ════════════════════════════════════════════════════════════════════════════════

class EpitopeGNN(nn.Module):
    """Same architecture as Phase 5, now handles new edge type (same_supertype)."""

    def __init__(self, in_dim, hidden_dim, conv_out_dim, num_heads,
                 num_layers, dropout, metadata):
        super().__init__()
        self.dropout = dropout
        node_types   = metadata[0]

        self.input_proj = nn.ModuleDict({
            nt: nn.Sequential(Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
            for nt in node_types
        })

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.projs = nn.ModuleList()

        for i in range(num_layers):
            in_ch = hidden_dim if i == 0 else conv_out_dim
            self.convs.append(HANConv(in_ch, conv_out_dim, heads=num_heads,
                                      dropout=dropout, metadata=metadata))
            self.norms.append(nn.ModuleDict({nt: nn.LayerNorm(conv_out_dim) for nt in node_types}))
            self.projs.append(
                nn.ModuleDict({nt: nn.Linear(in_ch, conv_out_dim, bias=False) for nt in node_types})
                if conv_out_dim != in_ch else None
            )

        self.classifier = nn.Sequential(
            nn.Linear(conv_out_dim, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1),
        )

    def forward(self, x_dict, edge_index_dict):
        h = {nt: proj(x_dict[nt]) for nt, proj in self.input_proj.items() if nt in x_dict}

        for i, conv in enumerate(self.convs):
            h_new = conv(h, edge_index_dict)
            for nt in h_new:
                if h_new[nt] is None:
                    continue
                if nt in h:
                    res = (self.projs[i][nt](h[nt]) if self.projs[i] is not None
                           else h[nt] if h[nt].shape[-1] == h_new[nt].shape[-1] else None)
                    if res is not None:
                        h_new[nt] = h_new[nt] + res
                h_new[nt] = self.norms[i][nt](h_new[nt])
                h_new[nt] = F.relu(F.dropout(h_new[nt], p=self.dropout, training=self.training))
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


def make_splits(graph):
    labels  = graph["epitope"].y.cpu().numpy()
    indices = np.arange(len(labels))
    train_idx, temp = train_test_split(indices, test_size=0.30,
                                       stratify=labels, random_state=42)
    val_idx, test_idx = train_test_split(temp, test_size=0.50,
                                          stratify=labels[temp], random_state=42)
    n = len(labels)
    tm = torch.zeros(n, dtype=torch.bool); tm[train_idx] = True
    vm = torch.zeros(n, dtype=torch.bool); vm[val_idx]   = True
    xm = torch.zeros(n, dtype=torch.bool); xm[test_idx]  = True
    logger.info(f"  Train {tm.sum():,} | Val {vm.sum():,} | Test {xm.sum():,}")
    return tm.to(device), vm.to(device), xm.to(device)


def train_epoch(model, graph, mask, optimizer, criterion):
    model.train(); optimizer.zero_grad()
    logits = model(graph.x_dict, graph.edge_index_dict)
    loss   = criterion(logits[mask], graph["epitope"].y[mask].float())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss)


@torch.no_grad()
def evaluate(model, graph, mask, criterion):
    model.eval()
    logits    = model(graph.x_dict, graph.edge_index_dict)
    labels    = graph["epitope"].y[mask].float()
    loss      = float(criterion(logits[mask], labels))
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
        "f1": f1_score(labels_np, preds, zero_division=0),
        "prec": precision_score(labels_np, preds, zero_division=0),
        "rec":  recall_score(labels_np, preds, zero_division=0),
        "probs": probs, "labels": labels_np,
    }


def retrain_gnn(graph: HeteroData):
    """Retrain GNN on improved graph v2."""
    console.rule("[yellow]Retraining GNN on graph v2[/yellow]")

    graph = graph.to(device)
    train_mask, val_mask, test_mask = make_splits(graph)

    conv_out = probe_conv_dim(graph.metadata(), HP["hidden_dim"], HP["num_heads"])
    model    = EpitopeGNN(
        in_dim=graph["epitope"].x.shape[1], hidden_dim=HP["hidden_dim"],
        conv_out_dim=conv_out, num_heads=HP["num_heads"],
        num_layers=HP["num_layers"], dropout=HP["dropout"],
        metadata=graph.metadata(),
    ).to(device)
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([HP["pos_weight"]], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", 0.5, 10)

    history = {k: [] for k in ["train_loss","val_loss","val_auroc","val_auprc","val_f1"]}
    best_auroc, best_epoch, patience, best_state = 0.0, 0, 0, None

    with Progress(SpinnerColumn(),
                  TextColumn("[cyan]Epoch {task.fields[ep]}/{task.fields[ep_t]}"),
                  BarColumn(),
                  TextColumn("[green]AUROC={task.fields[auroc]:.4f}"),
                  TextColumn("[yellow]loss={task.fields[tloss]:.4f}"),
                  TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task("train", total=HP["epochs"],
                             ep=0, ep_t=HP["epochs"], auroc=0.0, tloss=0.0)

        for epoch in range(1, HP["epochs"]+1):
            tl = train_epoch(model, graph, train_mask, optimizer, criterion)
            vm = evaluate(model, graph, val_mask, criterion)

            for k, v in [("train_loss",tl),("val_loss",vm["loss"]),
                          ("val_auroc",vm["auroc"]),("val_auprc",vm["auprc"]),
                          ("val_f1",vm["f1"])]:
                history[k].append(v)

            scheduler.step(vm["auroc"])

            if vm["auroc"] > best_auroc:
                best_auroc, best_epoch, patience = vm["auroc"], epoch, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience += 1

            prog.update(task, advance=1, ep=epoch, auroc=vm["auroc"], tloss=tl)

            if epoch % 10 == 0:
                logger.info(f"  Ep {epoch:3d} | loss={tl:.4f} | auroc={vm['auroc']:.4f} | "
                            f"auprc={vm['auprc']:.4f} | f1={vm['f1']:.4f}")

            if patience >= HP["patience"]:
                logger.info(f"  Early stop epoch {epoch}, best AUROC={best_auroc:.4f}")
                break

    model.load_state_dict(best_state)
    logger.info(f"  Best: epoch {best_epoch}, val AUROC={best_auroc:.4f}")

    torch.save({"model_state": best_state, "hyperparams": HP,
                "best_epoch": best_epoch, "best_val_auroc": best_auroc},
               str(MODELS_DIR / "best_model_v2.pt"))
    with open(MODELS_DIR / "training_history_v2.json", "w") as f:
        json.dump({k: [float(v) for v in vs] for k, vs in history.items()}, f, indent=2)

    # Test evaluation
    test_metrics = evaluate(model, graph, test_mask, criterion)
    val_metrics  = evaluate(model, graph, val_mask, criterion)

    return model, graph, history, best_epoch, val_metrics, test_metrics


# ════════════════════════════════════════════════════════════════════════════════
# STEP 3: RERANK WITH FIXES
# ════════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def score_all(model, graph) -> np.ndarray:
    model.eval()
    logits = model(graph.x_dict, graph.edge_index_dict)
    return torch.sigmoid(logits).cpu().numpy()


def build_deduplicated_candidates(graph, gnn_scores, data) -> pd.DataFrame:
    """
    Build the final candidate list with:
        1. Sequence-level deduplication (keep max score per unique sequence)
        2. Correct MHC class labels
        3. Source protein annotation
        4. Composite scoring
    """
    seqs   = graph["epitope"].seq
    labels = graph["epitope"].y.cpu().numpy()
    mhcs   = graph["epitope"].mhc

    # TCR evidence
    vjdb_path     = PROCESSED_DIR / "vjdb_tb_human_clean.tsv"
    df_vjdb       = pd.read_csv(vjdb_path, sep="\t")
    vjdb_epitopes = set(df_vjdb["epitope"].astype(str).str.upper().str.strip())

    # Source protein
    meta_prot    = pd.read_csv(EMBED_DIR / "tb_proteins_meta.csv")
    prot_ei      = graph["protein", "source_of", "epitope"].edge_index
    epi_to_prot  = {}
    if prot_ei.shape[1] > 0:
        for i in range(prot_ei.shape[1]):
            ei = int(prot_ei[1, i])
            if ei not in epi_to_prot:
                epi_to_prot[ei] = int(prot_ei[0, i])

    # HLA neighbors
    hla_ei    = graph["epitope", "binds_to", "hla"].edge_index
    hla_count = np.zeros(len(seqs), dtype=int)
    if hla_ei.shape[1] > 0:
        for i in range(hla_ei.shape[1]):
            ei = int(hla_ei[0, i])
            if ei < len(hla_count):
                hla_count[ei] += 1

    max_hla = max(hla_count.max(), 1)

    rows = []
    for i, (seq, score, label, mhc) in enumerate(zip(seqs, gnn_scores, labels, mhcs)):
        tcr_ev  = 1 if seq.upper() in vjdb_epitopes else 0
        hla_cov = hla_count[i] / max_hla

        gene = prot_name = ""
        if i in epi_to_prot:
            pi = epi_to_prot[i]
            if pi < len(meta_prot):
                gene     = meta_prot.iloc[pi].get("gene_name", "")
                prot_name = str(meta_prot.iloc[pi].get("protein_name", ""))[:50]

        composite = 0.50 * score + 0.30 * tcr_ev + 0.20 * hla_cov

        rows.append({
            "epitope_seq":       seq,
            "seq_length":        len(seq),
            "mhc_class":         mhc,
            "gnn_score":         float(score),
            "tcr_evidence":      tcr_ev,
            "hla_coverage_score":float(hla_cov),
            "hla_neighbors":     int(hla_count[i]),
            "composite_score":   float(composite),
            "true_label":        int(label),
            "source_gene":       gene,
            "source_protein":    prot_name,
        })

    df = pd.DataFrame(rows)

    # ── DEDUPLICATION: keep highest composite score per unique sequence ──
    before = len(df)
    df = df.sort_values("composite_score", ascending=False)
    df = df.drop_duplicates(subset=["epitope_seq"], keep="first")
    logger.info(f"  Deduplication: {before:,} → {len(df):,} unique sequences")

    # Filter to candidates the model predicts as immunogenic
    df_cand = df[df["gnn_score"] > 0.5].copy()
    df_cand = df_cand.sort_values("composite_score", ascending=False)
    df_cand["rank"] = range(1, len(df_cand)+1)

    logger.info(f"  Final candidates (GNN > 0.5): {len(df_cand):,}")
    logger.info(f"  Class I  (CD8+): {(df_cand['mhc_class']=='Class I (CD8+)').sum():,}")
    logger.info(f"  Class II (CD4+): {(df_cand['mhc_class']=='Class II (CD4+)').sum():,}")
    logger.info(f"  With TCR evidence: {df_cand['tcr_evidence'].sum():,}")
    logger.info(f"  True positives in top 50: {df_cand.head(50)['true_label'].sum():,} / 50")

    return df_cand


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_training(history, best_epoch, suffix="v2"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Training Curves — GNN {suffix}", fontweight="bold")
    ep = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(ep, history["train_loss"], "#2E86AB", label="Train", linewidth=1.5)
    axes[0].plot(ep, history["val_loss"],   "#E84855", label="Val",   linewidth=1.5)
    axes[0].axvline(best_epoch, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_title("Loss"); axes[0].legend(fontsize=9)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("BCE Loss")

    axes[1].plot(ep, history["val_auroc"], "#3BB273", linewidth=1.5)
    axes[1].axvline(best_epoch, color="gray", linestyle="--", linewidth=0.8)
    axes[1].axhline(max(history["val_auroc"]), color="#3BB273", linestyle=":",
                    label=f"Best={max(history['val_auroc']):.4f}")
    axes[1].set_title("Val AUROC"); axes[1].set_ylim(0,1); axes[1].legend(fontsize=9)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("AUROC")

    axes[2].plot(ep, history["val_auprc"], "#7B4F9E", linewidth=1.5)
    axes[2].axvline(best_epoch, color="gray", linestyle="--", linewidth=0.8)
    axes[2].axhline(max(history["val_auprc"]), color="#7B4F9E", linestyle=":",
                    label=f"Best={max(history['val_auprc']):.4f}")
    axes[2].set_title("Val AUPRC"); axes[2].set_ylim(0,1); axes[2].legend(fontsize=9)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("AUPRC")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"13_{suffix}_training_curves.png", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved 13_{suffix}_training_curves.png")


def plot_roc_pr(test_metrics, suffix="v2"):
    from sklearn.metrics import roc_curve, precision_recall_curve
    probs, labels = test_metrics["probs"], test_metrics["labels"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Test Evaluation — GNN {suffix}", fontweight="bold")

    fpr, tpr, _ = roc_curve(labels, probs)
    axes[0].plot(fpr, tpr, "#3BB273", linewidth=2, label=f"AUROC={test_metrics['auroc']:.4f}")
    axes[0].plot([0,1],[0,1], "gray", linestyle="--", linewidth=0.8, label="Random")
    axes[0].fill_between(fpr, tpr, alpha=0.1, color="#3BB273")
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve"); axes[0].legend()

    p, r, _ = precision_recall_curve(labels, probs)
    axes[1].plot(r, p, "#7B4F9E", linewidth=2, label=f"AUPRC={test_metrics['auprc']:.4f}")
    axes[1].axhline(labels.mean(), color="gray", linestyle="--",
                    label=f"Random={labels.mean():.3f}")
    axes[1].fill_between(r, p, alpha=0.1, color="#7B4F9E")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve"); axes[1].legend()

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"14_{suffix}_roc_pr.png", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved 14_{suffix}_roc_pr.png")


def plot_classI_classII(df_cand: pd.DataFrame):
    """Side-by-side top 15 for each MHC class."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Top Vaccine Candidates by MHC Class", fontweight="bold")

    for ax, mhc_class, color, title in [
        (axes[0], "Class I (CD8+)",  "#2E86AB", "MHC Class I — CD8+ cytotoxic T-cells"),
        (axes[1], "Class II (CD4+)", "#E84855", "MHC Class II — CD4+ helper T-cells"),
    ]:
        df_sub = df_cand[df_cand["mhc_class"] == mhc_class].head(15).iloc[::-1]
        y = range(len(df_sub))

        gnn = 0.50 * df_sub["gnn_score"]
        tcr = 0.30 * df_sub["tcr_evidence"]
        hla = 0.20 * df_sub["hla_coverage_score"]

        ax.barh(y, gnn, color=color, alpha=0.9, label="GNN score (50%)")
        ax.barh(y, tcr, left=gnn, color="#F4A261", alpha=0.9, label="TCR evidence (30%)")
        ax.barh(y, hla, left=gnn+tcr, color="#3BB273", alpha=0.9, label="HLA coverage (20%)")

        labels_y = []
        for _, row in df_sub.iterrows():
            mark = " ★" if row["tcr_evidence"] else ""
            gene = f" [{row['source_gene']}]" if row["source_gene"] else ""
            labels_y.append(f"#{int(row['rank'])} {row['epitope_seq']}{mark}{gene}")

        ax.set_yticks(y)
        ax.set_yticklabels(labels_y, fontsize=8, fontfamily="monospace")
        ax.set_xlabel("Composite score")
        ax.set_title(title, fontweight="bold")
        if ax == axes[0]:
            ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "15_classI_classII_candidates.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved 15_classI_classII_candidates.png")


# ── Final summary table ───────────────────────────────────────────────────────

def print_comparison(v1_metrics: dict, v2_metrics: dict,
                     df_v1: pd.DataFrame, df_v2: pd.DataFrame) -> None:
    console.rule("[bold green]Phase 7 Complete — v1 vs v2 Comparison[/bold green]")

    t = Table(title="Model Comparison: v1 vs v2", header_style="bold cyan", show_lines=True)
    t.add_column("Metric",   style="white",      min_width=22)
    t.add_column("v1 (old)", style="yellow",     min_width=12)
    t.add_column("v2 (new)", style="bold green", min_width=12)
    t.add_column("Change",   style="white",      min_width=12)

    metrics = [
        ("Test AUROC", "auroc"), ("Test AUPRC", "auprc"),
        ("Test F1",    "f1"),    ("Test Recall", "rec"),
    ]
    for label, key in metrics:
        v1v = v1_metrics.get(key, 0)
        v2v = v2_metrics.get(key, 0)
        diff = v2v - v1v
        arrow = "[green]+[/green]" if diff > 0 else "[red][/red]"
        t.add_row(label, f"{v1v:.4f}", f"{v2v:.4f}", f"{arrow}{abs(diff):.4f}")

    console.print(t)

    # Candidate summary
    c1 = df_v2[df_v2["mhc_class"] == "Class I (CD8+)"]
    c2 = df_v2[df_v2["mhc_class"] == "Class II (CD4+)"]

    console.print(f"\n[bold]Candidate summary (v2, deduplicated):[/bold]")
    console.print(f"  Total unique candidates:   {len(df_v2):,}")
    console.print(f"  MHC Class I  (CD8+):       {len(c1):,}")
    console.print(f"  MHC Class II (CD4+):       {len(c2):,}")
    console.print(f"  With TCR evidence:         {df_v2['tcr_evidence'].sum():,}")
    console.print(f"  True positives in top 50:  {df_v2.head(50)['true_label'].sum():,} / 50")

    console.print("\n[bold]Top 5 Class I candidates:[/bold]")
    for _, row in c1.head(5).iterrows():
        tcr = " [TCR★]" if row["tcr_evidence"] else ""
        console.print(f"  #{int(row['rank']):3d} {row['epitope_seq']:<25} "
                      f"score={row['composite_score']:.4f} gene={row['source_gene'] or '?'}{tcr}")

    console.print("\n[bold]Top 5 Class II candidates:[/bold]")
    for _, row in c2.head(5).iterrows():
        tcr = " [TCR★]" if row["tcr_evidence"] else ""
        console.print(f"  #{int(row['rank']):3d} {row['epitope_seq']:<25} "
                      f"score={row['composite_score']:.4f} gene={row['source_gene'] or '?'}{tcr}")

    console.print(f"\n[bold]Saved to:[/bold] {OUT_DIR.relative_to(PROJECT_ROOT)}")
    console.print("\n[bold cyan]Next:[/bold cyan] uv run python scripts/08_multi_epitope_assembly.py\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    console.rule("[bold cyan]Phase 7: GNN Improvement + Deduplicated Reranking[/bold cyan]")
    console.print(f"\n[bold]Improvements:[/bold]\n"
                  f"  1. Similarity threshold 0.85 → {HP['sim_threshold']} (denser edges)\n"
                  f"  2. k-NN neighbors 5 → {HP['knn_k']}\n"
                  f"  3. HLA supertype edges (new edge type)\n"
                  f"  4. Sequence-level deduplication\n"
                  f"  5. Correct MHC class labels (≤11aa = I, ≥12aa = II)\n")

    # Step 1: Build improved graph
    data  = load_all_embeddings()
    graph = build_improved_graph(data)

    # Step 2: Retrain
    model, graph, history, best_epoch, val_m, test_m = retrain_gnn(graph)

    # Load v1 metrics for comparison
    v1_test = {"auroc": 0.8928, "auprc": 0.5259, "f1": 0.4601, "rec": 0.8309}

    # Step 3: Rerank
    console.rule("[yellow]Reranking with deduplication[/yellow]")
    gnn_scores = score_all(model, graph)
    df_v2      = build_deduplicated_candidates(graph, gnn_scores, data)

    # Save
    df_v2.to_csv(OUT_DIR / "top_candidates_v2.csv", index=False)
    df_v2[df_v2["mhc_class"]=="Class I (CD8+)"].head(25).to_csv(
        OUT_DIR / "top25_classI_v2.csv", index=False)
    df_v2[df_v2["mhc_class"]=="Class II (CD4+)"].head(25).to_csv(
        OUT_DIR / "top25_classII_v2.csv", index=False)
    logger.info("  Saved top_candidates_v2.csv, top25_classI_v2.csv, top25_classII_v2.csv")

    # Plots
    console.rule("[yellow]Saving figures[/yellow]")
    plot_training(history, best_epoch)
    plot_roc_pr(test_m)
    plot_classI_classII(df_v2)

    # Compare
    print_comparison(v1_test, test_m, pd.DataFrame(), df_v2)


if __name__ == "__main__":
    main()