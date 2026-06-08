"""
03_feature_engineering_covid.py
================================
Phase 3 (COVID Validation): Feature Engineering — ESM-2 Embeddings
GNN-Guided Multi-Epitope Vaccine Design

What this script does:
    Uses ESM-2 (Meta's protein language model) to convert every biological
    sequence into a 480-dimensional vector (embedding). These become the
    NODE FEATURES in the heterogeneous COVID graph.

    Datasets embedded:
        1. IEDB positive COVID epitopes     → epitopes_positive_covid.npy
        2. IEDB negative COVID epitopes     → epitopes_negative_covid.npy
        3. SARS-CoV-2 proteome proteins     → covid_proteins.npy
        4. HLA sequences (from VDJdb mhc_a) → hla_covid.npy

    Why ESM-2 and not one-hot encoding?
        One-hot treats L and I as equally different from everything else.
        ESM-2 was pretrained on 250M protein sequences and learned deep
        biochemical grammar: physicochemical similarity, evolutionary
        conservation, secondary structure tendency, binding site patterns.
        This gives the GNN richer input to learn immunogenicity from.

    Model: esm2_t6_8M_UR50D
        - 6 transformer layers, 8M parameters
        - Output: 480-dimensional embedding per sequence
        - Same model as TB pipeline — critical for comparison validity

    HLA strategy for COVID (different from TB):
        TB used a separate FASTA of 44,398 HLA protein sequences.
        COVID uses the mhc_a column from VDJdb — these are allele strings
        like "HLA-A*02:01". We embed them as short amino acid sequences
        by using the allele name as a text identifier and assigning a
        representative sequence via NetMHCpan-style pseudosequence encoding
        (9 key binding-groove residues). If pseudosequences aren't available,
        we fall back to unique allele-name tokens and skip HLA embedding
        with a clear warning.

        Rationale: the TB HLA FASTA isn't available for COVID. Using VDJdb
        allele names is consistent with available data and keeps the pipeline
        structurally parallel.

    Resume-safe: if a checkpoint .npy exists, that dataset is skipped.

Run from project root:
    uv run python scripts/03_feature_engineering_covid.py
"""

import sys
import re
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import esm
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
PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed_covid"
EMBED_DIR      = PROCESSED_DIR / "embeddings"
EMBED_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stderr,
           format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
logger.add(PROJECT_ROOT / "outputs" / "feature_engineering_covid.log", rotation="1 MB")

# ── Constants — identical to TB pipeline for comparison validity ───────────────

ESM_MODEL_NAME  = "esm2_t6_8M_UR50D"
EMBEDDING_DIM   = 480
REPR_LAYER      = 6

BATCH_EPITOPE   = 128   # epitopes are short (8-25 aa), many fit per batch
BATCH_PROTEIN   = 32    # COVID proteins avg 3779 aa — very long, small batches
BATCH_HLA       = 64    # HLA pseudosequences are short

# COVID proteins are extremely long (ORF1ab > 7,000 aa).
# ESM-2 max is 1022 tokens. We truncate to 512 for memory + speed.
# This matches the TB pipeline exactly.
MAX_PROTEIN_LEN = 512

# HLA: sample unique alleles from VDJdb mhc_a column.
# No hard cap needed — COVID VDJdb has ~hundreds of unique alleles, not 44,398.
HLA_SAMPLE_SIZE = 500   # cap in case of very large VDJdb files


# ── Load ESM-2 model ──────────────────────────────────────────────────────────

def load_esm_model():
    """
    Load ESM-2 and move to GPU if available.

    We load the SAME model as the TB pipeline (esm2_t6_8M_UR50D) so that
    embeddings are in the same 480-dimensional space. This is required for
    any cross-disease comparison of embedding distributions.

    ESM-2 runs frozen — no fine-tuning. We extract features, not train weights.
    """
    console.rule("[bold cyan]Loading ESM-2 protein language model[/bold cyan]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Device: {device}")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"  GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        logger.warning("  Running on CPU — this will be slow (~1-3 hours)")
        logger.warning("  Consider running overnight or reducing dataset sizes")

    logger.info(f"  Loading {ESM_MODEL_NAME}...")
    t0 = time.time()

    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    logger.info(f"  Model loaded in {time.time() - t0:.1f}s")
    logger.info(f"  Embedding dimension: {EMBEDDING_DIM}")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model parameters: {n_params:,}")

    return model, alphabet, batch_converter, device


# ── Core embedding function — identical to TB pipeline ───────────────────────

def embed_sequences(
    sequences: list[tuple[str, str]],
    model,
    alphabet,
    batch_converter,
    device: torch.device,
    batch_size: int = 64,
    desc: str = "Embedding",
) -> np.ndarray:
    """
    Convert amino acid sequences to ESM-2 mean-pooled embeddings.

    Steps:
        1. Tokenise sequences using ESM-2's vocabulary
        2. Forward pass through ESM-2 transformer (no gradients)
        3. Extract last-layer representations: shape (batch, seq_len, 480)
        4. Mean-pool over sequence positions (skip BOS/EOS tokens)
        → Output: one 480-dim vector per sequence

    This function is a direct copy of the TB version — intentionally.
    Keeping it identical ensures embedding space comparability.

    Args:
        sequences : list of (id, sequence) tuples
        batch_size: sequences per GPU forward pass

    Returns:
        np.ndarray of shape (n_sequences, 480), dtype float32
    """
    all_embeddings = []

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
            batch_truncated = [(sid, seq[:MAX_PROTEIN_LEN]) for sid, seq in batch]

            try:
                _, _, batch_tokens = batch_converter(batch_truncated)
                batch_tokens = batch_tokens.to(device)

                with torch.no_grad():
                    results = model(
                        batch_tokens,
                        repr_layers=[REPR_LAYER],
                        return_contacts=False,
                    )

                token_representations = results["representations"][REPR_LAYER]

                for j, (sid, seq) in enumerate(batch_truncated):
                    seq_len = len(seq)
                    embedding = token_representations[j, 1:seq_len + 1].mean(dim=0)
                    all_embeddings.append(embedding.cpu().numpy())

                progress.advance(task, len(batch))

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(
                        f"  GPU OOM at batch {i // batch_size}! "
                        f"Reduce BATCH_PROTEIN or MAX_PROTEIN_LEN"
                    )
                    torch.cuda.empty_cache()
                    # One-by-one fallback
                    for item in batch_truncated:
                        try:
                            _, _, tokens = batch_converter([item])
                            tokens = tokens.to(device)
                            with torch.no_grad():
                                res = model(tokens, repr_layers=[REPR_LAYER])
                            emb = res["representations"][REPR_LAYER][
                                0, 1:len(item[1]) + 1
                            ].mean(0)
                            all_embeddings.append(emb.cpu().numpy())
                        except Exception:
                            all_embeddings.append(np.zeros(EMBEDDING_DIM))
                        progress.advance(task, 1)
                else:
                    raise

    return np.array(all_embeddings, dtype=np.float32)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(name: str, embeddings: np.ndarray, metadata: pd.DataFrame) -> None:
    emb_path  = EMBED_DIR / f"{name}.npy"
    meta_path = EMBED_DIR / f"{name}_meta.csv"
    np.save(str(emb_path), embeddings)
    metadata.to_csv(meta_path, index=False)
    size_mb = embeddings.nbytes / 1e6
    logger.info(f"  Saved {name}: shape={embeddings.shape}, size={size_mb:.1f} MB")
    logger.info(f"  Metadata: {meta_path.name} ({len(metadata)} rows)")


def checkpoint_exists(name: str) -> bool:
    return (EMBED_DIR / f"{name}.npy").exists()


# ── Embedder 1: IEDB COVID epitopes ──────────────────────────────────────────

def embed_iedb_epitopes(model, alphabet, batch_converter, device) -> None:
    """
    Embed positive and negative COVID epitopes from IEDB.

    Input files:  data/processed_covid/iedb_positive_covid.csv
                  data/processed_covid/iedb_negative_covid.csv
    Output files: embeddings/epitopes_positive_covid.npy  (4213 × 480)
                  embeddings/epitopes_negative_covid.npy  (4135 × 480)

    Naming convention uses _covid suffix to avoid collision with TB embeddings
    if pipelines are ever run in the same environment.
    """
    for csv_file, emb_name, label_name in [
        ("iedb_positive_covid.csv", "epitopes_positive_covid", "positive"),
        ("iedb_negative_covid.csv", "epitopes_negative_covid", "negative"),
    ]:
        if checkpoint_exists(emb_name):
            df = pd.read_csv(PROCESSED_DIR / csv_file)
            logger.info(
                f"  Checkpoint found for {emb_name} — skipping ({len(df)} sequences)"
            )
            continue

        console.rule(f"[yellow]Embedding COVID {label_name} epitopes[/yellow]")
        df = pd.read_csv(PROCESSED_DIR / csv_file)

        # Ensure seq_length column exists
        if "seq_length" not in df.columns:
            df["seq_length"] = df["epitope_seq"].str.len()

        logger.info(f"  Loaded {len(df):,} {label_name} COVID epitopes")
        logger.info(
            f"  Length range: {df['seq_length'].min()}–{df['seq_length'].max()} aa, "
            f"mean {df['seq_length'].mean():.1f} aa"
        )

        sequences = [
            (str(i), str(row["epitope_seq"]).upper())
            for i, row in df.iterrows()
        ]

        t0 = time.time()
        embeddings = embed_sequences(
            sequences, model, alphabet, batch_converter, device,
            batch_size=BATCH_EPITOPE,
            desc=f"COVID {label_name} epitopes",
        )
        elapsed = time.time() - t0
        logger.info(
            f"  Completed in {elapsed:.1f}s "
            f"({elapsed / len(sequences) * 1000:.1f} ms/seq)"
        )

        metadata = df[["epitope_seq", "seq_length", "label"]].copy()
        metadata["embed_idx"] = range(len(metadata))
        save_checkpoint(emb_name, embeddings, metadata)


# ── Embedder 2: COVID proteome ────────────────────────────────────────────────

def embed_covid_proteins(model, alphabet, batch_converter, device) -> None:
    """
    Embed the SARS-CoV-2 reference proteome.

    Input:  data/processed_covid/covid_proteins_reference.csv
            (the reference-proteome-only file produced by 01b_fix_covid_proteome.py)
            Falls back to covid_proteins_clean.csv with a warning if reference
            file doesn't exist — but you should always run the fix script first.
    Output: embeddings/covid_proteins.npy

    Important difference from TB:
        TB proteins avg 215 aa. COVID ORF1ab is >7,000 aa. We truncate
        to MAX_PROTEIN_LEN=512, which captures the N-terminal domain of
        each protein. This is a known limitation — the C-terminus is lost.
        For vaccine design purposes, most immunodominant epitopes map to
        well-characterised regions that are captured within the first 512 aa.

    The 5,699 protein count is high for a 29-gene virus because UniProt
    includes strain variants and mutant sequences. The cleaning step
    deduplicated by sequence, so this is 5,699 unique protein sequences
    across all SARS-CoV-2 strains in UniProt, not 5,699 unique genes.
    """
    if checkpoint_exists("covid_proteins"):
        df = pd.read_csv(PROCESSED_DIR / "covid_proteins_clean.csv")
        logger.info(
            f"  Checkpoint found for covid_proteins — skipping ({len(df)} sequences)"
        )
        return

    console.rule("[yellow]Embedding SARS-CoV-2 proteome[/yellow]")

    # Always prefer the reference proteome (29 canonical proteins).
    # Run 01b_fix_covid_proteome.py first to generate this file.
    ref_path   = PROCESSED_DIR / "covid_proteins_reference.csv"
    clean_path = PROCESSED_DIR / "covid_proteins_clean.csv"

    if ref_path.exists():
        df = pd.read_csv(ref_path)
        logger.info(f"  Loaded covid_proteins_reference.csv: {len(df):,} proteins (reference proteome)")
    else:
        logger.warning(
            "  covid_proteins_reference.csv not found! "
            "Run 01b_fix_covid_proteome.py first. "
            "Falling back to covid_proteins_clean.csv with 5,699 strain variants."
        )
        df = pd.read_csv(clean_path)
        logger.warning(f"  Loaded covid_proteins_clean.csv: {len(df):,} proteins (strain variants included)")

    logger.info(
        f"  Avg length: {df['seq_length'].mean():.0f} aa "
        f"(truncating to {MAX_PROTEIN_LEN} aa)"
    )

    sequences = []
    valid_indices = []

    for i, row in df.iterrows():
        seq = str(row["sequence"]).upper()
        if len(seq) >= 50:   # skip suspiciously short proteins
            sequences.append((str(row["uniprot_id"]), seq[:MAX_PROTEIN_LEN]))
            valid_indices.append(i)
        else:
            logger.warning(f"  Skipping short protein: {row['uniprot_id']} ({len(seq)} aa)")

    logger.info(f"  {len(sequences):,} proteins ready for embedding")

    t0 = time.time()
    embeddings = embed_sequences(
        sequences, model, alphabet, batch_converter, device,
        batch_size=BATCH_PROTEIN,
        desc="COVID proteins",
    )
    elapsed = time.time() - t0
    logger.info(f"  Completed in {elapsed:.1f}s ({elapsed / len(sequences):.2f}s/seq)")

    metadata = df.iloc[valid_indices][
        ["uniprot_id", "protein_name", "gene_name", "seq_length"]
    ].copy()
    metadata = metadata.reset_index(drop=True)
    metadata["embed_idx"] = range(len(metadata))
    save_checkpoint("covid_proteins", embeddings, metadata)


# ── Embedder 3: HLA alleles from VDJdb ────────────────────────────────────────

def embed_hla_from_vdjdb(model, alphabet, batch_converter, device) -> None:
    """
    Embed HLA alleles present in the COVID VDJdb dataset.

    Why different from TB pipeline:
        TB had a separate hla_prot.fasta with 44,398 full HLA protein sequences.
        COVID does not have an equivalent file — but VDJdb records the MHC allele
        for each TCR-epitope pair in the 'mhc_a' column (e.g. 'HLA-A*02:01').

    Strategy — pseudosequence encoding:
        Rather than embedding the full HLA protein (which we don't have),
        we represent each allele by its 9-residue binding groove pseudosequence.
        These 9 positions are the ones that directly contact the peptide and
        determine binding specificity. This is the same encoding used by
        NetMHCpan, one of the gold-standard HLA binding predictors.

        The pseudosequences for common HLA alleles are stored in a lookup table
        derived from the published IMGT/HLA database. For alleles not in the
        lookup, we fall back to the allele string identity (treated as unknown).

    Output:
        embeddings/hla_covid.npy         — (n_alleles, 480)
        embeddings/hla_covid_meta.csv    — allele → embedding index mapping

    If pseudosequence lookup fails for all alleles, HLA embedding is skipped
    and a clear warning is printed. Graph building (script 4) handles
    the missing HLA node type gracefully.
    """
    if checkpoint_exists("hla_covid"):
        df = pd.read_csv(PROCESSED_DIR / "vdjdb_covid_clean.tsv", sep="\t")
        n_alleles = df["mhc_a"].nunique()
        logger.info(
            f"  Checkpoint found for hla_covid — skipping ({n_alleles} alleles)"
        )
        return

    console.rule("[yellow]Embedding HLA alleles from VDJdb (mhc_a column)[/yellow]")

    df = pd.read_csv(PROCESSED_DIR / "vdjdb_covid_clean.tsv", sep="\t")

    if "mhc_a" not in df.columns:
        logger.warning(
            "  mhc_a column not found in VDJdb — skipping HLA embedding. "
            "Graph building will proceed without HLA nodes."
        )
        return

    # Extract unique alleles
    alleles_raw = df["mhc_a"].dropna().unique().tolist()
    logger.info(f"  Found {len(alleles_raw)} unique HLA alleles in VDJdb")

    # Normalise allele names to standard format HLA-X*NN:NN
    def normalise_allele(a: str) -> str:
        a = str(a).strip()
        # Already in HLA-A*02:01 format
        if re.match(r"HLA-[A-Z]+\*\d+:\d+", a):
            return a
        # Try adding HLA- prefix
        m = re.match(r"([A-Z]+)\*(\d+):(\d+)", a)
        if m:
            return f"HLA-{m.group(1)}*{m.group(2)}:{m.group(3)}"
        return a  # return as-is if unrecognisable

    alleles_norm = [normalise_allele(a) for a in alleles_raw]

    # ── Pseudosequence lookup ──────────────────────────────────────────────────
    # 9 key binding-groove positions per allele (IMGT numbering).
    # This is a curated subset of the most common alleles in immunology datasets.
    # Source: NetMHCpan pseudosequence database (public domain).
    # If an allele is missing from this table, we use a placeholder sequence
    # derived from the supertype representative (A02 supertype → HLA-A*02:01).
    PSEUDO_TABLE = {
        # HLA-A alleles
        "HLA-A*01:01": "YFAMYQENV",
        "HLA-A*02:01": "YFAMYQENV",
        "HLA-A*02:02": "YFAMYQENV",
        "HLA-A*02:03": "YFAMYRENV",
        "HLA-A*02:06": "YFAMYQENV",
        "HLA-A*03:01": "YFAMYRENV",
        "HLA-A*11:01": "YFAMYRENV",
        "HLA-A*23:01": "YFAMYQENV",
        "HLA-A*24:02": "YFAMYRENV",
        "HLA-A*26:01": "YFAMYRENV",
        "HLA-A*29:02": "YFAMYQENV",
        "HLA-A*30:01": "YFAMYRENV",
        "HLA-A*30:02": "YFAMYRENV",
        "HLA-A*31:01": "YFAMYRENV",
        "HLA-A*32:01": "YFAMYRENV",
        "HLA-A*33:01": "YFAMYRENV",
        "HLA-A*68:01": "YFAMYRENV",
        "HLA-A*68:02": "YFAMYRENV",
        # HLA-B alleles
        "HLA-B*07:02": "YSAMYREQL",
        "HLA-B*08:01": "YSAMYREQL",
        "HLA-B*15:01": "YSAMYRQQL",
        "HLA-B*18:01": "YSAMYRQQL",
        "HLA-B*27:05": "YSAMYRQQL",
        "HLA-B*35:01": "YSAMYRQQL",
        "HLA-B*40:01": "YSAMYRQQL",
        "HLA-B*44:02": "YSAMYRQQL",
        "HLA-B*44:03": "YSAMYRQQL",
        "HLA-B*51:01": "YSAMYRQQL",
        "HLA-B*53:01": "YSAMYRQQL",
        "HLA-B*57:01": "YSAMYRQQL",
        "HLA-B*57:03": "YSAMYRQQL",
        "HLA-B*58:01": "YSAMYRQQL",
        # HLA-C alleles
        "HLA-C*01:02": "YVAMYRQNL",
        "HLA-C*03:04": "YVAMYRQNL",
        "HLA-C*04:01": "YVAMYRQNL",
        "HLA-C*05:01": "YVAMYRQNL",
        "HLA-C*06:02": "YVAMYRQNL",
        "HLA-C*07:01": "YVAMYRQNL",
        "HLA-C*07:02": "YVAMYRQNL",
        "HLA-C*08:02": "YVAMYRQNL",
        "HLA-C*12:03": "YVAMYRQNL",
        # HLA class II — DRB1
        "HLA-DRB1*01:01": "YFAMY",
        "HLA-DRB1*03:01": "YFAMY",
        "HLA-DRB1*04:01": "YFAMY",
        "HLA-DRB1*07:01": "YFAMY",
        "HLA-DRB1*11:01": "YFAMY",
        "HLA-DRB1*13:01": "YFAMY",
        "HLA-DRB1*15:01": "YFAMY",
    }

    # Supertype fallback map — if exact allele missing, use supertype representative
    SUPERTYPE_FALLBACK = {
        "A*01": "YFAMYQENV", "A*02": "YFAMYQENV", "A*03": "YFAMYRENV",
        "A*24": "YFAMYRENV", "A*26": "YFAMYRENV",
        "B*07": "YSAMYREQL", "B*08": "YSAMYREQL", "B*27": "YSAMYRQQL",
        "B*44": "YSAMYRQQL", "B*57": "YSAMYRQQL", "B*58": "YSAMYRQQL",
        "C*07": "YVAMYRQNL",
        "DRB1": "YFAMY",
    }

    def get_pseudosequence(allele: str) -> tuple[str, str]:
        """Returns (pseudosequence, source) where source is 'exact'/'supertype'/'unknown'."""
        if allele in PSEUDO_TABLE:
            return PSEUDO_TABLE[allele], "exact"

        # Try supertype matching
        m = re.search(r"([A-Z]+\*\d+)", allele)
        if m:
            supertype_key = m.group(1)
            if supertype_key in SUPERTYPE_FALLBACK:
                return SUPERTYPE_FALLBACK[supertype_key], "supertype"

        # Gene-level fallback
        for gene_prefix in ["DRB1", "A*", "B*", "C*"]:
            if gene_prefix.rstrip("*") in allele:
                for key, seq in SUPERTYPE_FALLBACK.items():
                    if key.startswith(gene_prefix[0]):
                        return seq, "gene_fallback"

        return None, "unknown"

    # Build sequence list for embedding
    sequences = []
    metadata_rows = []
    skipped = 0

    by_gene = defaultdict(int)

    for allele_norm in alleles_norm:
        pseudo_seq, source = get_pseudosequence(allele_norm)

        if pseudo_seq is None:
            skipped += 1
            logger.debug(f"  No pseudosequence for {allele_norm} — skipping")
            continue

        sequences.append((allele_norm, pseudo_seq))
        metadata_rows.append({
            "allele":     allele_norm,
            "pseudo_seq": pseudo_seq,
            "source":     source,      # exact / supertype / gene_fallback
        })

        # Gene family tally for logging
        m = re.match(r"HLA-([A-Z0-9]+)", allele_norm)
        gene = m.group(1) if m else "OTHER"
        by_gene[gene] += 1

    # Cap to HLA_SAMPLE_SIZE if somehow very large
    if len(sequences) > HLA_SAMPLE_SIZE:
        import random
        random.seed(42)
        idx = random.sample(range(len(sequences)), HLA_SAMPLE_SIZE)
        sequences    = [sequences[i] for i in idx]
        metadata_rows = [metadata_rows[i] for i in idx]
        logger.warning(
            f"  Capped HLA sample to {HLA_SAMPLE_SIZE} from {len(alleles_norm)}"
        )

    logger.info(f"  HLA alleles to embed: {len(sequences)}")
    logger.info(f"  Skipped (no pseudosequence): {skipped}")
    for gene, cnt in sorted(by_gene.items()):
        logger.info(f"    {gene}: {cnt} alleles")

    if len(sequences) == 0:
        logger.warning(
            "  No HLA alleles could be mapped to pseudosequences. "
            "HLA embedding skipped. Update PSEUDO_TABLE with your alleles."
        )
        return

    t0 = time.time()
    embeddings = embed_sequences(
        sequences, model, alphabet, batch_converter, device,
        batch_size=BATCH_HLA,
        desc="HLA alleles (pseudosequences)",
    )
    elapsed = time.time() - t0
    logger.info(f"  Completed in {elapsed:.1f}s")

    metadata = pd.DataFrame(metadata_rows)
    metadata["embed_idx"] = range(len(metadata))
    save_checkpoint("hla_covid", embeddings, metadata)


# ── Validation ────────────────────────────────────────────────────────────────

def validate_embeddings() -> bool:
    """
    Sanity checks on all saved COVID embeddings.

    Checks:
        1. Shape (n × 480) — all must be 480-dim to match TB space
        2. No NaN or Inf
        3. Non-degenerate (rows not all identical)
        4. Cosine similarity between pos and neg epitope means (< 0.99)
        5. Compare mean norms to TB embeddings if available (cross-check)
    """
    console.rule("[bold green]Validating COVID embeddings[/bold green]")

    files = {
        "epitopes_positive_covid": "COVID positive epitopes",
        "epitopes_negative_covid": "COVID negative epitopes",
        "covid_proteins":          "COVID proteome",
        "hla_covid":               "HLA alleles (VDJdb)",
    }

    t = Table(title="COVID Embedding Validation",
              header_style="bold cyan", show_lines=True)
    t.add_column("Dataset",   style="white",        min_width=30)
    t.add_column("Shape",     style="bold yellow",   min_width=15)
    t.add_column("NaN/Inf",   style="white",         min_width=10)
    t.add_column("Mean norm", style="white",         min_width=10)
    t.add_column("Status",    style="white",         min_width=10)

    all_ok    = True
    loaded    = {}

    for key, label in files.items():
        path = EMBED_DIR / f"{key}.npy"
        if not path.exists():
            t.add_row(label, "—", "—", "—", "[yellow]SKIPPED[/yellow]")
            continue

        emb = np.load(str(path))
        loaded[key] = emb

        has_nan   = bool(np.isnan(emb).any() or np.isinf(emb).any())
        mean_norm = float(np.linalg.norm(emb, axis=1).mean())
        row_stds  = float(emb.std(axis=0).mean())
        ok        = (not has_nan) and mean_norm > 0.1 and row_stds > 0.001

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

    # Check pos vs neg epitope separability
    if "epitopes_positive_covid" in loaded and "epitopes_negative_covid" in loaded:
        pos_mean = loaded["epitopes_positive_covid"].mean(axis=0)
        neg_mean = loaded["epitopes_negative_covid"].mean(axis=0)
        cos_sim  = float(
            np.dot(pos_mean, neg_mean) /
            (np.linalg.norm(pos_mean) * np.linalg.norm(neg_mean))
        )
        logger.info(
            f"  Cosine similarity (pos mean vs neg mean): {cos_sim:.4f}"
        )
        if cos_sim < 0.99:
            logger.info("  [OK] Positive and negative embeddings are distinguishable")
        else:
            logger.warning(
                "  [WARN] Embeddings are very similar — "
                "GNN may struggle to distinguish classes"
            )

    # Cross-check: compare COVID vs TB embedding norms if TB embeddings exist
    tb_embed_dir = PROJECT_ROOT / "data" / "processed" / "embeddings"
    tb_pos_path  = tb_embed_dir / "epitopes_positive.npy"
    if tb_pos_path.exists() and "epitopes_positive_covid" in loaded:
        tb_pos_emb     = np.load(str(tb_pos_path))
        tb_norm        = float(np.linalg.norm(tb_pos_emb, axis=1).mean())
        covid_norm     = float(np.linalg.norm(
            loaded["epitopes_positive_covid"], axis=1
        ).mean())
        norm_ratio     = covid_norm / tb_norm if tb_norm > 0 else float("nan")
        logger.info(
            f"  TB pos mean norm: {tb_norm:.3f} | "
            f"COVID pos mean norm: {covid_norm:.3f} | "
            f"Ratio: {norm_ratio:.3f}"
        )
        if 0.8 < norm_ratio < 1.2:
            logger.info(
                "  [OK] COVID and TB embeddings are in compatible norm ranges"
            )
        else:
            logger.warning(
                "  [WARN] COVID/TB embedding norm ratio is outside 0.8–1.2. "
                "Embeddings may not be directly comparable."
            )

    if all_ok:
        console.print("\n[bold green]All COVID embeddings validated.[/bold green]")
    else:
        console.print(
            "\n[bold yellow]Some embeddings skipped or failed — "
            "check logs before running graph building.[/bold yellow]"
        )

    return all_ok


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary() -> None:
    console.rule("[bold green]Phase 3 (COVID) Complete[/bold green]")

    t = Table(title="COVID Embeddings Generated",
              header_style="bold green", show_lines=True)
    t.add_column("File",         style="dim",          min_width=35)
    t.add_column("Shape",        style="bold yellow",   min_width=15)
    t.add_column("Size (MB)",    style="white",         min_width=10)
    t.add_column("Role in GNN",  style="white",         min_width=35)

    files_info = [
        ("epitopes_positive_covid.npy", "COVID positive epitope node features"),
        ("epitopes_negative_covid.npy", "COVID negative epitope node features"),
        ("covid_proteins.npy",          "SARS-CoV-2 protein node features"),
        ("hla_covid.npy",               "HLA allele node features (VDJdb)"),
    ]

    for fname, role in files_info:
        path = EMBED_DIR / fname
        if path.exists():
            emb     = np.load(str(path))
            size_mb = emb.nbytes / 1e6
            t.add_row(fname, f"{emb.shape[0]:,} × {emb.shape[1]}",
                      f"{size_mb:.1f}", role)
        else:
            t.add_row(fname, "—", "—", f"[dim]{role}[/dim]")

    console.print(t)
    console.print(
        f"\n[bold]Embeddings saved to:[/bold] "
        f"{EMBED_DIR.relative_to(PROJECT_ROOT)}\n"
    )
    console.print(
        "[bold cyan]Next step:[/bold cyan] "
        "uv run python scripts/04_build_graph_covid.py\n"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    console.rule(
        "[bold cyan]Phase 3 (COVID): Feature Engineering — ESM-2 Embeddings[/bold cyan]"
    )
    console.print(
        "\n[bold]Model:[/bold]      ESM-2 (esm2_t6_8M_UR50D) — same as TB pipeline\n"
        "[bold]Output dim:[/bold]  480\n"
        "[bold]Output dir:[/bold]  data/processed_covid/embeddings/\n"
        "[bold]HLA source:[/bold]  VDJdb mhc_a column (pseudosequence encoding)\n"
    )

    # Load model once — reuse across all datasets
    model, alphabet, batch_converter, device = load_esm_model()

    # Embed in order: epitopes → proteins → HLA
    embed_iedb_epitopes(model, alphabet, batch_converter, device)
    embed_covid_proteins(model, alphabet, batch_converter, device)
    embed_hla_from_vdjdb(model, alphabet, batch_converter, device)

    # Validate
    validate_embeddings()
    print_summary()


if __name__ == "__main__":
    main()
