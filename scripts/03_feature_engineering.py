"""
03_feature_engineering.py
==========================
Phase 3: Feature Engineering — Protein Language Model Embeddings
GNN-Guided Multi-Epitope Vaccine Design

What this script does:
    Uses ESM-2 (Meta's protein language model) to convert every biological
    sequence into a 480-dimensional vector (embedding). These vectors become
    the NODE FEATURES in our heterogeneous graph.

    Why ESM-2 and not just one-hot encoding?
        One-hot encoding treats each amino acid independently — it has no
        concept that Leucine (L) and Isoleucine (I) are biochemically similar.
        ESM-2 was trained on 250 million protein sequences and learned the
        deep biochemical "grammar" of proteins. Its embeddings capture:
            - Amino acid physicochemical properties
            - Evolutionary conservation
            - Secondary structure tendency
            - Binding site characteristics
        This gives the GNN vastly richer information to learn from.

    Model used: esm2_t6_8M_UR50D
        - 6 transformer layers, 8 million parameters
        - Output: 480-dimensional embedding per sequence
        - Fast enough for 21,000+ sequences on laptop GPU
        - Good quality for short peptides (8-25 aa)

    Outputs (data/processed/embeddings/):
        epitopes_positive.npy   — shape (2249,  480)
        epitopes_negative.npy   — shape (21635, 480)
        tb_proteins.npy         — shape (21008, 480)
        hla_sample.npy          — shape (N,     480)
        *.csv files             — metadata (sequence ID → embedding row index)

    Resume-safe: if a checkpoint exists it skips already-processed batches.

Run from project root:
    uv run python scripts/03_feature_engineering.py
"""

import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import esm
from Bio import SeqIO
from loguru import logger
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn,
    TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn,
)
from rich.table import Table

# ── Setup ─────────────────────────────────────────────────────────────────────

console = Console()

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed"
EMBED_DIR      = PROCESSED_DIR / "embeddings"
EMBED_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
logger.add(PROJECT_ROOT / "outputs" / "feature_engineering.log", rotation="1 MB")

# ── Constants ─────────────────────────────────────────────────────────────────

# ESM-2 model — 8M parameter version, outputs 480-dim embeddings
# Small enough to fit in 6 GB VRAM with room to spare
ESM_MODEL_NAME  = "esm2_t6_8M_UR50D"
EMBEDDING_DIM   = 480
REPR_LAYER      = 6        # which transformer layer to extract (last = most semantic)

# Batch sizes — tuned for RTX 4050 6GB VRAM
# Epitopes are short (8-25 aa) so we can fit many per batch
# Proteins are longer (avg 215 aa) so smaller batches
BATCH_EPITOPE   = 128
BATCH_PROTEIN   = 32
BATCH_HLA       = 64

# For HLA: we have 44,398 alleles but many are very similar
# We sample a representative subset to save time and disk space
HLA_SAMPLE_SIZE = 2000

# Maximum protein length to embed (very long proteins get truncated)
# ESM-2 can handle up to 1022 tokens; we truncate at 512 for speed
MAX_PROTEIN_LEN = 512

# ── Load ESM-2 model ──────────────────────────────────────────────────────────

def load_esm_model():
    """
    Load ESM-2 and move to GPU.

    ESM-2 is a transformer model pretrained on protein sequences.
    We use it as a FROZEN feature extractor — we don't fine-tune it,
    we just use its learned representations as input features.

    This is called "transfer learning" — using knowledge learned on
    a large dataset (250M proteins) for our specific task (TB epitopes).
    """
    console.rule("[bold cyan]Loading ESM-2 protein language model[/bold cyan]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Device: {device}")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"  GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        logger.warning("  Running on CPU — this will be slow (~2-4 hours)")
        logger.warning("  Consider running overnight or reducing dataset sizes")

    logger.info(f"  Loading {ESM_MODEL_NAME}...")
    t0 = time.time()

    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model = model.eval().to(device)   # eval mode = no dropout, no gradient tracking
    batch_converter = alphabet.get_batch_converter()

    logger.info(f"  Model loaded in {time.time() - t0:.1f}s")
    logger.info(f"  Embedding dimension: {EMBEDDING_DIM}")

    # Count parameters for the paper
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model parameters: {n_params:,}")

    return model, alphabet, batch_converter, device


# ── Core embedding function ───────────────────────────────────────────────────

def embed_sequences(
    sequences: list[tuple[str, str]],   # list of (id, sequence) tuples
    model,
    alphabet,
    batch_converter,
    device: torch.device,
    batch_size: int = 64,
    desc: str = "Embedding",
) -> np.ndarray:
    """
    Convert a list of amino acid sequences into ESM-2 embeddings.

    Process:
        1. Convert sequences to token IDs (ESM's vocabulary)
        2. Pass tokens through ESM-2 transformer
        3. Extract the representation from the last layer
        4. Mean-pool over sequence length → one 480-dim vector per sequence

    Why mean pooling?
        Sequences have different lengths. Mean pooling averages the
        per-position representations into a single fixed-size vector.
        This is the standard approach for sequence-level tasks.

    Args:
        sequences: list of (id, sequence) tuples
        batch_size: how many sequences to process at once (GPU memory limited)

    Returns:
        numpy array of shape (n_sequences, 480)
    """
    all_embeddings = []
    n_batches = (len(sequences) + batch_size - 1) // batch_size

    with Progress(
        SpinnerColumn(),
        TextColumn(f"[cyan]{desc}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(desc, total=len(sequences))

        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]

            # Truncate very long sequences to MAX_PROTEIN_LEN
            batch_truncated = [(sid, seq[:MAX_PROTEIN_LEN]) for sid, seq in batch]

            try:
                # Convert to ESM token format
                _, _, batch_tokens = batch_converter(batch_truncated)
                batch_tokens = batch_tokens.to(device)

                with torch.no_grad():   # no gradients needed — we're just extracting features
                    results = model(
                        batch_tokens,
                        repr_layers=[REPR_LAYER],
                        return_contacts=False,
                    )

                # Extract per-token representations: shape (batch, seq_len, 480)
                token_representations = results["representations"][REPR_LAYER]

                # Mean pool over sequence positions (skip BOS and EOS tokens)
                for j, (sid, seq) in enumerate(batch_truncated):
                    seq_len = len(seq)
                    # Tokens: [BOS, aa1, aa2, ..., aaN, EOS]
                    # We pool positions 1 to seq_len (inclusive), skipping BOS/EOS
                    embedding = token_representations[j, 1:seq_len + 1].mean(dim=0)
                    all_embeddings.append(embedding.cpu().numpy())

                progress.advance(task, len(batch))

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"  GPU OOM at batch {i//batch_size}! "
                                 f"Reduce batch_size or MAX_PROTEIN_LEN")
                    torch.cuda.empty_cache()
                    # Try processing one by one as fallback
                    for item in batch_truncated:
                        try:
                            _, _, tokens = batch_converter([item])
                            tokens = tokens.to(device)
                            with torch.no_grad():
                                res = model(tokens, repr_layers=[REPR_LAYER])
                            emb = res["representations"][REPR_LAYER][0, 1:len(item[1])+1].mean(0)
                            all_embeddings.append(emb.cpu().numpy())
                        except Exception:
                            # Absolute fallback: zero vector
                            all_embeddings.append(np.zeros(EMBEDDING_DIM))
                        progress.advance(task, 1)
                else:
                    raise

    return np.array(all_embeddings, dtype=np.float32)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(name: str, embeddings: np.ndarray, metadata: pd.DataFrame) -> None:
    """Save embeddings + metadata. Called after each dataset completes."""
    emb_path  = EMBED_DIR / f"{name}.npy"
    meta_path = EMBED_DIR / f"{name}_meta.csv"
    np.save(str(emb_path), embeddings)
    metadata.to_csv(meta_path, index=False)
    size_mb = embeddings.nbytes / 1e6
    logger.info(f"  Saved {name}: shape={embeddings.shape}, size={size_mb:.1f} MB")
    logger.info(f"  Metadata: {meta_path.name} ({len(metadata)} rows)")


def checkpoint_exists(name: str) -> bool:
    """Check if this embedding has already been computed."""
    return (EMBED_DIR / f"{name}.npy").exists()


# ── Embedder functions ────────────────────────────────────────────────────────

def embed_iedb_epitopes(model, alphabet, batch_converter, device) -> None:
    """
    Embed all IEDB epitopes (positive + negative).

    Each epitope is a short peptide (8-25 aa).
    Output: two .npy files, one per label class.
    """
    for split, filename, label_name in [
        ("iedb_positive_clean.csv", "epitopes_positive", "positive"),
        ("iedb_negative_clean.csv", "epitopes_negative", "negative"),
    ]:
        if checkpoint_exists(filename):
            df = pd.read_csv(PROCESSED_DIR / split)
            logger.info(f"  Checkpoint found for {filename} — skipping ({len(df)} sequences)")
            continue

        console.rule(f"[yellow]Embedding IEDB {label_name} epitopes[/yellow]")
        df = pd.read_csv(PROCESSED_DIR / split)
        logger.info(f"  Loaded {len(df):,} {label_name} epitopes")

        sequences = [
            (str(i), row["epitope_seq"])
            for i, row in df.iterrows()
        ]

        t0 = time.time()
        embeddings = embed_sequences(
            sequences, model, alphabet, batch_converter, device,
            batch_size=BATCH_EPITOPE,
            desc=f"IEDB {label_name}",
        )
        elapsed = time.time() - t0
        logger.info(f"  Completed in {elapsed:.1f}s ({elapsed/len(sequences)*1000:.1f}ms/seq)")

        metadata = df[["epitope_seq", "seq_length", "label"]].copy()
        metadata["embed_idx"] = range(len(metadata))
        save_checkpoint(filename, embeddings, metadata)


def embed_tb_proteins(model, alphabet, batch_converter, device) -> None:
    """
    Embed the full M. tuberculosis H37Rv proteome.

    Proteins are much longer than epitopes (avg 215 aa, max 512 after truncation).
    We use smaller batches to fit in VRAM.

    These embeddings become the PROTEIN NODES in the heterogeneous graph.
    """
    if checkpoint_exists("tb_proteins"):
        df = pd.read_csv(PROCESSED_DIR / "tb_proteome_metadata.csv")
        logger.info(f"  Checkpoint found for tb_proteins — skipping ({len(df)} sequences)")
        return

    console.rule("[yellow]Embedding TB proteome[/yellow]")
    df = pd.read_csv(PROCESSED_DIR / "tb_proteome_metadata.csv")
    logger.info(f"  Loaded {len(df):,} TB proteins")
    logger.info(f"  Truncating sequences to {MAX_PROTEIN_LEN} aa for memory efficiency")

    # Parse sequences from FASTA (metadata CSV has IDs, FASTA has sequences)
    fasta_path = PROCESSED_DIR / "tb_proteome_clean.fasta"
    seq_dict = {
        rec.id: str(rec.seq)[:MAX_PROTEIN_LEN]
        for rec in SeqIO.parse(str(fasta_path), "fasta")
    }
    logger.info(f"  Parsed {len(seq_dict):,} sequences from FASTA")

    # Match metadata to sequences
    sequences = []
    valid_indices = []
    for i, row in df.iterrows():
        uid = row["uniprot_id"]
        if uid in seq_dict:
            sequences.append((uid, seq_dict[uid]))
            valid_indices.append(i)
        else:
            logger.warning(f"  No sequence found for {uid}")

    logger.info(f"  {len(sequences):,} proteins ready for embedding")

    t0 = time.time()
    embeddings = embed_sequences(
        sequences, model, alphabet, batch_converter, device,
        batch_size=BATCH_PROTEIN,
        desc="TB proteins",
    )
    elapsed = time.time() - t0
    logger.info(f"  Completed in {elapsed:.1f}s ({elapsed/len(sequences):.2f}s/seq)")

    metadata = df.iloc[valid_indices][["uniprot_id", "gene_name", "protein_name", "seq_length"]].copy()
    metadata = metadata.reset_index(drop=True)
    metadata["embed_idx"] = range(len(metadata))
    save_checkpoint("tb_proteins", embeddings, metadata)


def embed_hla_sequences(model, alphabet, batch_converter, device) -> None:
    """
    Embed a representative sample of HLA protein sequences.

    We have 44,398 HLA alleles but many are extremely similar
    (differing by only 1-2 amino acids). Embedding all of them would take
    ~2 hours and produce mostly redundant information.

    Strategy: sample HLA_SAMPLE_SIZE alleles, stratified by gene family
    (proportional sampling so A/B/C/DR are all represented).

    These become the HLA NODES in the heterogeneous graph.
    """
    if checkpoint_exists("hla_sample"):
        logger.info(f"  Checkpoint found for hla_sample — skipping")
        return

    console.rule("[yellow]Embedding HLA sequences (stratified sample)[/yellow]")

    import re
    hla_path = PROCESSED_DIR / "hla_prot_clean.fasta"
    all_records = list(SeqIO.parse(str(hla_path), "fasta"))
    logger.info(f"  Loaded {len(all_records):,} HLA sequences")

    # Assign gene family to each record
    records_with_gene = []
    for rec in all_records:
        m = re.search(r"\b([A-Z0-9]+)\*\d+:\d+", rec.description.upper())
        gene = m.group(1) if m else "OTHER"
        records_with_gene.append((gene, rec))

    # Stratified sampling: keep proportional representation per gene
    from collections import defaultdict
    import random
    random.seed(42)

    gene_groups = defaultdict(list)
    for gene, rec in records_with_gene:
        gene_groups[gene].append(rec)

    sampled_records = []
    total = len(all_records)
    for gene, recs in gene_groups.items():
        n_sample = max(1, round(HLA_SAMPLE_SIZE * len(recs) / total))
        sampled = random.sample(recs, min(n_sample, len(recs)))
        sampled_records.extend([(gene, r) for r in sampled])
        logger.info(f"  HLA-{gene}: {len(recs):,} total → {len(sampled)} sampled")

    # Trim to exactly HLA_SAMPLE_SIZE
    random.shuffle(sampled_records)
    sampled_records = sampled_records[:HLA_SAMPLE_SIZE]
    logger.info(f"  Final sample: {len(sampled_records)} HLA sequences")

    sequences = [(rec.id, str(rec.seq)[:MAX_PROTEIN_LEN]) for _, rec in sampled_records]
    genes     = [gene for gene, _ in sampled_records]

    t0 = time.time()
    embeddings = embed_sequences(
        sequences, model, alphabet, batch_converter, device,
        batch_size=BATCH_HLA,
        desc="HLA sequences",
    )
    elapsed = time.time() - t0
    logger.info(f"  Completed in {elapsed:.1f}s")

    metadata = pd.DataFrame({
        "hla_id":    [rec.id for _, rec in sampled_records],
        "gene":      genes,
        "allele":    [rec.description for _, rec in sampled_records],
        "embed_idx": range(len(sampled_records)),
    })
    save_checkpoint("hla_sample", embeddings, metadata)


# ── Validation ────────────────────────────────────────────────────────────────

def validate_embeddings() -> None:
    """
    Quick sanity checks on the saved embeddings.

    Checks:
        1. Shape is correct (n_sequences × 480)
        2. No NaN or Inf values
        3. Embeddings are not all identical (would mean something went wrong)
        4. Positive and negative epitope embeddings are distinguishable
    """
    console.rule("[bold green]Validating embeddings[/bold green]")

    files = {
        "epitopes_positive": "Positive epitopes",
        "epitopes_negative": "Negative epitopes",
        "tb_proteins":       "TB proteins",
        "hla_sample":        "HLA sequences",
    }

    t = Table(title="Embedding Validation", header_style="bold cyan", show_lines=True)
    t.add_column("Dataset",     style="white")
    t.add_column("Shape",       style="bold yellow")
    t.add_column("NaN/Inf",     style="white")
    t.add_column("Mean norm",   style="white")
    t.add_column("Status",      style="white")

    all_ok = True
    loaded = {}

    for key, label in files.items():
        path = EMBED_DIR / f"{key}.npy"
        if not path.exists():
            t.add_row(label, "—", "—", "—", "[red]MISSING[/red]")
            all_ok = False
            continue

        emb = np.load(str(path))
        loaded[key] = emb

        has_nan = np.isnan(emb).any() or np.isinf(emb).any()
        mean_norm = float(np.linalg.norm(emb, axis=1).mean())

        # Check embeddings aren't all identical
        row_stds = emb.std(axis=0).mean()
        ok = not has_nan and mean_norm > 0.1 and row_stds > 0.001

        t.add_row(
            label,
            f"{emb.shape[0]:,} × {emb.shape[1]}",
            "[red]YES[/red]" if has_nan else "[green]None[/green]",
            f"{mean_norm:.3f}",
            "[green]OK[/green]" if ok else "[red]FAIL[/red]",
        )
        if not ok:
            all_ok = False

    console.print(t)

    # Extra check: are positive and negative epitope embeddings different?
    if "epitopes_positive" in loaded and "epitopes_negative" in loaded:
        pos_mean = loaded["epitopes_positive"].mean(axis=0)
        neg_mean = loaded["epitopes_negative"].mean(axis=0)
        cosine_sim = float(
            np.dot(pos_mean, neg_mean) /
            (np.linalg.norm(pos_mean) * np.linalg.norm(neg_mean))
        )
        logger.info(f"  Cosine similarity (pos mean vs neg mean): {cosine_sim:.4f}")
        if cosine_sim < 0.99:
            logger.info("  Positive and negative embeddings are distinguishable")
        else:
            logger.warning("  Embeddings are very similar — may indicate an issue")

    if all_ok:
        console.print("\n[bold green]All embeddings validated successfully![/bold green]")
    else:
        console.print("\n[bold red]Some embeddings failed validation — check logs[/bold red]")

    return all_ok


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary() -> None:
    console.rule("[bold green]Phase 3 Complete[/bold green]")

    t = Table(title="Generated Embeddings", header_style="bold green")
    t.add_column("File",         style="dim")
    t.add_column("Shape",        style="bold yellow")
    t.add_column("Size",         style="white")
    t.add_column("Role in GNN",  style="white")

    files_info = [
        ("epitopes_positive.npy", "Positive epitope node features"),
        ("epitopes_negative.npy", "Negative epitope node features"),
        ("tb_proteins.npy",       "Protein node features"),
        ("hla_sample.npy",        "HLA node features"),
    ]

    for fname, role in files_info:
        path = EMBED_DIR / fname
        if path.exists():
            emb = np.load(str(path))
            size_mb = emb.nbytes / 1e6
            t.add_row(fname, f"{emb.shape[0]:,} × {emb.shape[1]}",
                      f"{size_mb:.1f} MB", role)

    console.print(t)
    console.print(f"\n[bold]All embeddings saved to:[/bold] {EMBED_DIR.relative_to(PROJECT_ROOT)}")
    console.print(
        "\n[bold cyan]Next step:[/bold cyan] "
        "uv run python scripts/04_build_graph.py\n"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    console.rule("[bold cyan]Phase 3: Feature Engineering (ESM-2 Embeddings)[/bold cyan]")
    console.print(
        "\n[bold]Model:[/bold] ESM-2 (esm2_t6_8M_UR50D)\n"
        "[bold]Output dim:[/bold] 480\n"
        "[bold]Output dir:[/bold] data/processed/embeddings/\n"
    )

    # Load model once — reuse for all datasets
    model, alphabet, batch_converter, device = load_esm_model()

    # Embed each dataset
    embed_iedb_epitopes(model, alphabet, batch_converter, device)
    embed_tb_proteins(model, alphabet, batch_converter, device)
    embed_hla_sequences(model, alphabet, batch_converter, device)

    # Validate all outputs
    validate_embeddings()
    print_summary()


if __name__ == "__main__":
    main()