"""
05_train_gnn.py
===============
Phase 5: Train the Heterogeneous Graph Neural Network
GNN-Guided Multi-Epitope Vaccine Design

Architecture note for PyG 2.7 HANConv:
    HANConv in PyG 2.7 AVERAGES attention heads (not concatenates).
    So HANConv(in=D, out=D, heads=H) → output shape (N, D).
    This means hidden_dim stays constant through all layers: simple and clean.

Dimension flow:
    Input (320) → Linear → (128) → HANConv → (128) → HANConv → (128) → MLP → (1)

Run from project root:
    uv run python scripts/05_train_gnn.py
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv, Linear
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, average_precision_score,
    confusion_matrix,
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAPH_DIR    = PROJECT_ROOT / "data" / "processed" / "graph"
MODELS_DIR   = PROJECT_ROOT / "outputs" / "models"
FIGURES_DIR  = PROJECT_ROOT / "outputs" / "figures"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
)
logger.add(PROJECT_ROOT / "outputs" / "training.log", rotation="5 MB")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")
if device.type == "cuda":
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Hyperparameters ───────────────────────────────────────────────────────────

HP = {
    # PyG 2.7 HANConv averages heads → output dim = out_channels (not * heads)
    # So we simply set hidden_dim throughout and let HANConv handle attention internally
    "hidden_dim":  128,
    "num_heads":   4,      # used internally by HANConv for attention diversity
    "num_layers":  3,
    "dropout":     0.3,
    "lr":          1e-3,
    "weight_decay":1e-4,
    "epochs":      200,
    "patience":    20,
    "pos_weight":  9.6,
    "train_ratio": 0.70,
    "val_ratio":   0.15,
    "test_ratio":  0.15,
    "random_seed": 42,
}

# ── Probe HANConv output dim ──────────────────────────────────────────────────

def probe_hanconv_output_dim(metadata: tuple, hidden_dim: int, num_heads: int) -> int:
    """
    PyG's HANConv output dim depends on the version — some concatenate,
    some average. We probe it empirically with a dummy forward pass
    so the rest of the architecture is always correct regardless of version.
    """
    import torch
    dummy_x    = {nt: torch.zeros(2, hidden_dim) for nt in metadata[0]}
    # Build a minimal valid edge_index_dict (empty edges)
    dummy_ei   = {et: torch.zeros(2, 0, dtype=torch.long) for et in metadata[1]}
    try:
        conv   = HANConv(hidden_dim, hidden_dim, heads=num_heads, metadata=metadata)
        out    = conv(dummy_x, dummy_ei)
        # Get output dim from first node type that has output
        for nt in metadata[0]:
            if nt in out and out[nt] is not None:
                actual_dim = out[nt].shape[1]
                logger.info(f"  HANConv probe: in={hidden_dim}, heads={num_heads} "
                            f"→ out={actual_dim} (averaging={'yes' if actual_dim==hidden_dim else 'no, concatenating'})")
                return actual_dim
    except Exception as e:
        logger.warning(f"  HANConv probe failed ({e}), assuming averaging → dim={hidden_dim}")
    return hidden_dim


# ── GNN Architecture ──────────────────────────────────────────────────────────

class EpitopeGNN(nn.Module):
    """
    Heterogeneous Attention Network for epitope immunogenicity prediction.

    Uses a probed conv_out_dim so it works regardless of whether your
    version of PyG concatenates or averages attention heads.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        conv_out_dim: int,     # actual HANConv output dim (probed at runtime)
        num_heads: int,
        num_layers: int,
        dropout: float,
        metadata: tuple,
    ):
        super().__init__()
        self.dropout     = dropout
        node_types       = metadata[0]
        self.conv_out_dim = conv_out_dim

        # ── Input projection: in_dim → hidden_dim ──
        self.input_proj = nn.ModuleDict({
            nt: nn.Sequential(
                Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
            for nt in node_types
        })

        # ── HANConv layers ──
        # Layer 0: hidden_dim → conv_out_dim
        # Layer 1+: conv_out_dim → conv_out_dim
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.projs = nn.ModuleList()   # project conv_out_dim back to hidden_dim if needed

        for i in range(num_layers):
            in_ch = hidden_dim if i == 0 else conv_out_dim
            self.convs.append(
                HANConv(
                    in_channels=in_ch,
                    out_channels=conv_out_dim,
                    heads=num_heads,
                    dropout=dropout,
                    metadata=metadata,
                )
            )
            self.norms.append(nn.ModuleDict({
                nt: nn.LayerNorm(conv_out_dim) for nt in node_types
            }))
            # If conv_out_dim != hidden_dim, project back for residual
            if conv_out_dim != in_ch:
                self.projs.append(nn.ModuleDict({
                    nt: nn.Linear(in_ch, conv_out_dim, bias=False) for nt in node_types
                }))
            else:
                self.projs.append(None)

        # ── Classifier: conv_out_dim → 64 → 1 ──
        self.classifier = nn.Sequential(
            nn.Linear(conv_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x_dict: dict, edge_index_dict: dict) -> torch.Tensor:
        # Step 1: project inputs
        h = {}
        for nt, proj in self.input_proj.items():
            if nt in x_dict:
                h[nt] = proj(x_dict[nt])

        # Step 2: message passing
        for i, conv in enumerate(self.convs):
            h_new = conv(h, edge_index_dict)

            for nt in h_new:
                if h_new[nt] is None:
                    continue

                # Residual connection (with projection if dims differ)
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

        # Step 3: classify epitopes
        return self.classifier(h["epitope"]).squeeze(-1)


# ── Data ──────────────────────────────────────────────────────────────────────

def load_graph() -> HeteroData:
    path  = GRAPH_DIR / "heterogeneous_graph.pt"
    graph = torch.load(str(path), map_location=device, weights_only=False)
    logger.info(f"Loaded graph: {graph.node_types}, {len(graph.edge_types)} edge types")
    return graph


def make_splits(graph: HeteroData):
    labels  = graph["epitope"].y.cpu().numpy()
    indices = np.arange(len(labels))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=HP["val_ratio"] + HP["test_ratio"],
        stratify=labels,
        random_state=HP["random_seed"],
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=HP["test_ratio"] / (HP["val_ratio"] + HP["test_ratio"]),
        stratify=labels[temp_idx],
        random_state=HP["random_seed"],
    )

    n = len(labels)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True

    logger.info(f"  Train {train_mask.sum():,} | Val {val_mask.sum():,} | Test {test_mask.sum():,}")
    logger.info(f"  Positives in train: {labels[train_idx].sum():,}")

    return train_mask.to(device), val_mask.to(device), test_mask.to(device)


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(model, graph, mask, optimizer, criterion) -> float:
    model.train()
    optimizer.zero_grad()
    logits = model(graph.x_dict, graph.edge_index_dict)
    loss   = criterion(logits[mask], graph["epitope"].y[mask].float())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return float(loss)


@torch.no_grad()
def evaluate(model, graph, mask, criterion) -> dict:
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
        "loss":   loss,
        "auroc":  auroc,
        "auprc":  auprc,
        "f1":     f1_score(labels_np, preds, zero_division=0),
        "prec":   precision_score(labels_np, preds, zero_division=0),
        "rec":    recall_score(labels_np, preds, zero_division=0),
        "probs":  probs,
        "labels": labels_np,
    }


def train(model, graph, train_mask, val_mask):
    console.rule("[yellow]Training[/yellow]")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([HP["pos_weight"]], device=device)
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )

    history = {k: [] for k in ["train_loss","val_loss","val_auroc","val_auprc","val_f1"]}
    best_auroc, best_epoch, patience_count, best_state = 0.0, 0, 0, None

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]Epoch {task.fields[ep]}/{task.fields[ep_total]}"),
        BarColumn(),
        TextColumn("[green]AUROC={task.fields[auroc]:.4f}"),
        TextColumn("[yellow]loss={task.fields[tloss]:.4f}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "train",
            total=HP["epochs"],
            ep=0, ep_total=HP["epochs"], auroc=0.0, tloss=0.0,
        )

        for epoch in range(1, HP["epochs"] + 1):
            tl  = train_epoch(model, graph, train_mask, optimizer, criterion)
            vm  = evaluate(model, graph, val_mask, criterion)

            history["train_loss"].append(tl)
            history["val_loss"].append(vm["loss"])
            history["val_auroc"].append(vm["auroc"])
            history["val_auprc"].append(vm["auprc"])
            history["val_f1"].append(vm["f1"])

            scheduler.step(vm["auroc"])

            if vm["auroc"] > best_auroc:
                best_auroc, best_epoch, patience_count = vm["auroc"], epoch, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_count += 1

            progress.update(task, advance=1, ep=epoch, auroc=vm["auroc"], tloss=tl)

            if epoch % 10 == 0:
                logger.info(
                    f"  Ep {epoch:3d} | loss={tl:.4f} | "
                    f"auroc={vm['auroc']:.4f} | auprc={vm['auprc']:.4f} | f1={vm['f1']:.4f}"
                )

            if patience_count >= HP["patience"]:
                logger.info(f"  Early stop at epoch {epoch}, best AUROC={best_auroc:.4f}")
                break

    model.load_state_dict(best_state)
    logger.info(f"  Best: epoch {best_epoch}, val AUROC={best_auroc:.4f}")

    torch.save(
        {"model_state": best_state, "hyperparams": HP,
         "best_epoch": best_epoch, "best_val_auroc": best_auroc},
        str(MODELS_DIR / "best_model.pt"),
    )
    with open(MODELS_DIR / "training_history.json", "w") as f:
        json.dump({k: [float(v) for v in vs] for k, vs in history.items()}, f, indent=2)

    return history, best_epoch


# ── Plots ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.family": "DejaVu Sans",
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "figure.facecolor": "white",
})


def plot_training_curves(history: dict, best_epoch: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Training Curves — Heterogeneous GNN", fontweight="bold")
    ep = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(ep, history["train_loss"], color="#2E86AB", label="Train", linewidth=1.5)
    axes[0].plot(ep, history["val_loss"],   color="#E84855", label="Val",   linewidth=1.5)
    axes[0].axvline(best_epoch, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_title("Loss"); axes[0].legend(fontsize=9)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("BCE Loss")

    axes[1].plot(ep, history["val_auroc"], color="#3BB273", linewidth=1.5)
    axes[1].axvline(best_epoch, color="gray", linestyle="--", linewidth=0.8)
    axes[1].axhline(max(history["val_auroc"]), color="#3BB273", linestyle=":",
                    label=f"Best={max(history['val_auroc']):.4f}")
    axes[1].set_title("Val AUROC"); axes[1].set_ylim(0, 1); axes[1].legend(fontsize=9)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("AUROC")

    axes[2].plot(ep, history["val_auprc"], color="#7B4F9E", linewidth=1.5)
    axes[2].axvline(best_epoch, color="gray", linestyle="--", linewidth=0.8)
    axes[2].axhline(max(history["val_auprc"]), color="#7B4F9E", linestyle=":",
                    label=f"Best={max(history['val_auprc']):.4f}")
    axes[2].set_title("Val AUPRC"); axes[2].set_ylim(0, 1); axes[2].legend(fontsize=9)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("AUPRC")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "08_training_curves.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved 08_training_curves.png")


def plot_roc_and_pr(test_metrics: dict) -> None:
    from sklearn.metrics import roc_curve, precision_recall_curve
    probs, labels = test_metrics["probs"], test_metrics["labels"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Test Set Evaluation — Heterogeneous GNN", fontweight="bold")

    fpr, tpr, _ = roc_curve(labels, probs)
    axes[0].plot(fpr, tpr, color="#3BB273", linewidth=2,
                 label=f"AUROC = {test_metrics['auroc']:.4f}")
    axes[0].plot([0,1],[0,1], color="gray", linestyle="--", linewidth=0.8, label="Random")
    axes[0].fill_between(fpr, tpr, alpha=0.1, color="#3BB273")
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve"); axes[0].legend(fontsize=10)

    prec_arr, rec_arr, _ = precision_recall_curve(labels, probs)
    axes[1].plot(rec_arr, prec_arr, color="#7B4F9E", linewidth=2,
                 label=f"AUPRC = {test_metrics['auprc']:.4f}")
    axes[1].axhline(labels.mean(), color="gray", linestyle="--", linewidth=0.8,
                    label=f"Random = {labels.mean():.3f}")
    axes[1].fill_between(rec_arr, prec_arr, alpha=0.1, color="#7B4F9E")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve"); axes[1].legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "09_roc_pr_curves.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved 09_roc_pr_curves.png")


# ── Final summary ─────────────────────────────────────────────────────────────

def print_final_results(val_metrics: dict, test_metrics: dict) -> None:
    console.rule("[bold green]Final Results[/bold green]")

    t = Table(title="Model Performance", header_style="bold cyan", show_lines=True)
    t.add_column("Metric",   style="white",      min_width=20)
    t.add_column("Val Set",  style="bold yellow", min_width=12)
    t.add_column("Test Set", style="bold green",  min_width=12)
    t.add_column("Notes",    style="dim",          min_width=35)

    metrics_info = [
        ("auroc", "AUROC",     "Main metric. >0.80 good, >0.90 excellent"),
        ("auprc", "AUPRC",     "Imbalanced data key. >0.40 good"),
        ("f1",    "F1 Score",  "Balance of precision + recall"),
        ("prec",  "Precision", "Of predicted positives, how many are real?"),
        ("rec",   "Recall",    "Of all positives, how many did we find?"),
    ]
    for key, label, note in metrics_info:
        t.add_row(label, f"{val_metrics[key]:.4f}", f"{test_metrics[key]:.4f}", note)

    console.print(t)

    cm = confusion_matrix(
        test_metrics["labels"],
        (test_metrics["probs"] >= 0.5).astype(int)
    )
    console.print("\n[bold]Test confusion matrix:[/bold]")
    console.print(f"  True Neg:  {cm[0,0]:,}  |  False Pos: {cm[0,1]:,}")
    console.print(f"  False Neg: {cm[1,0]:,}  |  True Pos:  {cm[1,1]:,}")
    console.print("\n[bold cyan]Next:[/bold cyan] uv run python scripts/06_prioritize_epitopes.py\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    console.rule("[bold cyan]Phase 5: Training Heterogeneous GNN[/bold cyan]")

    graph = load_graph().to(device)

    # Probe HANConv output dim for this PyG version before building model
    console.rule("[yellow]Probing HANConv output dimensions[/yellow]")
    conv_out_dim = probe_hanconv_output_dim(
        graph.metadata(), HP["hidden_dim"], HP["num_heads"]
    )
    console.print(
        f"\n[bold]HANConv:[/bold] in={HP['hidden_dim']}, "
        f"heads={HP['num_heads']} → out={conv_out_dim}\n"
        f"[bold]Device:[/bold] {device}\n"
    )

    console.rule("[yellow]Data splits[/yellow]")
    train_mask, val_mask, test_mask = make_splits(graph)

    console.rule("[yellow]Model[/yellow]")
    model = EpitopeGNN(
        in_dim       = graph["epitope"].x.shape[1],
        hidden_dim   = HP["hidden_dim"],
        conv_out_dim = conv_out_dim,
        num_heads    = HP["num_heads"],
        num_layers   = HP["num_layers"],
        dropout      = HP["dropout"],
        metadata     = graph.metadata(),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {n_params:,}")

    t0 = time.time()
    history, best_epoch = train(model, graph, train_mask, val_mask)
    logger.info(f"  Training time: {time.time()-t0:.1f}s")

    criterion    = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([HP["pos_weight"]], device=device)
    )
    val_metrics  = evaluate(model, graph, val_mask,  criterion)
    test_metrics = evaluate(model, graph, test_mask, criterion)

    plot_training_curves(history, best_epoch)
    plot_roc_and_pr(test_metrics)
    print_final_results(val_metrics, test_metrics)


if __name__ == "__main__":
    main()