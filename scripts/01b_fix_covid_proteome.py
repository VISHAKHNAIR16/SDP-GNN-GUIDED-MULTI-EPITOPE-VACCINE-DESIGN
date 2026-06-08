"""
01b_fix_covid_proteome.py
=========================
Pre-processing fix: Filter covid_proteins_clean.csv down to
SARS-CoV-2 reference proteome only (29 canonical proteins).

Problem:
    covid_proteins_clean.csv has 5,699 entries because the raw FASTA
    contained proteins from ALL UniProt SARS-CoV-2 strain submissions —
    Delta, Omicron, Alpha, Wuhan, etc. — not just the reference proteome.
    Deduplication by sequence still leaves thousands of near-identical
    strain variant sequences.

Why this matters:
    The heterogeneous graph maps protein → epitope edges.
    With 5,699 protein nodes for 29 genes, each epitope will connect to
    hundreds of nearly-identical protein nodes (one per strain variant),
    creating dense spurious edges that don't represent distinct biology.
    The GNN will learn strain-variant noise, not protein function.

Fix strategy (two-step, in order of preference):
    Step 1 — Gene-level deduplication:
        Keep only one protein per unique gene name (best sequence =
        longest, which tends to be the reference strain entry).
        This reduces 5,699 → ~29 entries covering all 29 genes.

    Step 2 — If gene names are missing/sparse:
        Keep the longest unique sequence per protein cluster
        (proteins with >95% sequence identity → keep only the longest).
        This is a fallback that doesn't require gene annotations.

Output:
    data/processed_covid/covid_proteins_reference.csv   ← use this in script 3
    (original covid_proteins_clean.csv is kept untouched as backup)

Run from project root:
    uv run python scripts/01b_fix_covid_proteome.py
"""

import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
from loguru import logger
from rich.console import Console
from rich.table import Table

# ── Setup ─────────────────────────────────────────────────────────────────────

console = Console()

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_covid"

logger.remove()
logger.add(sys.stderr,
           format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")


# ── Known SARS-CoV-2 canonical gene names ────────────────────────────────────

# These are the 29 proteins encoded by the SARS-CoV-2 reference genome
# (Wuhan-Hu-1, NC_045512.2 / UniProt UP000464024).
# Source: UniProt reference proteome + NCBI annotation.
CANONICAL_GENES = {
    # Structural proteins
    "S",    # Spike glycoprotein
    "E",    # Envelope small membrane protein
    "M",    # Membrane glycoprotein
    "N",    # Nucleoprotein (Nucleocapsid)
    # Non-structural proteins (from ORF1ab polyprotein cleavage)
    "ORF1ab", "NSP1", "NSP2", "NSP3", "NSP4", "NSP5", "NSP6",
    "NSP7",  "NSP8",  "NSP9",  "NSP10", "NSP12", "NSP13",
    "NSP14", "NSP15", "NSP16",
    # Accessory proteins
    "ORF3A", "ORF3B",
    "ORF6",
    "ORF7A", "ORF7B",
    "ORF8",
    "ORF9B",
    "ORF10",
    # Alternative names for ORF1ab components
    "RdRp",    # RNA-dependent RNA polymerase = NSP12
    "Helicase", # = NSP13
    "PLpro",   # Papain-like protease = part of NSP3
    "3CLpro",  # Main protease = NSP5
    "Mpro",    # = NSP5
}

# Normalise gene name variants to canonical form
GENE_NORMALISE = {
    "SPIKE": "S",
    "SURFACE GLYCOPROTEIN": "S",
    "NUCLEOCAPSID": "N",
    "NUCLEOPROTEIN": "N",
    "ENVELOPE": "E",
    "MEMBRANE": "M",
    "REPLICASE": "ORF1ab",
    "POLYPROTEIN": "ORF1ab",
    "ORF1A": "ORF1ab",
    "ORF1B": "ORF1ab",
    "NON-STRUCTURAL PROTEIN 12": "NSP12",
    "NON-STRUCTURAL PROTEIN 13": "NSP13",
    "RNA-DEPENDENT RNA POLYMERASE": "NSP12",
    "RDRP": "NSP12",
    "HELICASE": "NSP13",
    "MAIN PROTEASE": "NSP5",
    "3C-LIKE PROTEASE": "NSP5",
    "PAPAIN-LIKE PROTEASE": "NSP3",
    "ORF3": "ORF3A",
    "ORF7": "ORF7A",
    "ORF9": "ORF9B",
}


def normalise_gene(gene: str) -> str:
    """Normalise messy gene name to canonical form."""
    if not isinstance(gene, str) or not gene.strip():
        return ""
    g = gene.strip().upper()
    return GENE_NORMALISE.get(g, g)


# ── Step 1: Gene-level deduplication ─────────────────────────────────────────

def deduplicate_by_gene(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep one protein per unique gene name — the longest sequence,
    which is most likely to be the full-length reference entry.

    Returns a DataFrame of one row per gene.
    """
    logger.info("Step 1: Gene-level deduplication")

    df = df.copy()
    df["gene_norm"] = df["gene_name"].apply(normalise_gene)

    # Separate rows with recognised gene names from those without
    has_gene  = df[df["gene_norm"] != ""]
    no_gene   = df[df["gene_norm"] == ""]

    logger.info(f"  Rows with gene name   : {len(has_gene):,}")
    logger.info(f"  Rows without gene name: {len(no_gene):,}")

    # For each gene, keep the longest sequence
    best_per_gene = (
        has_gene
        .sort_values("seq_length", ascending=False)
        .drop_duplicates(subset=["gene_norm"], keep="first")
    )

    logger.info(f"  Unique genes after dedup: {len(best_per_gene)}")

    # Log which genes we found
    genes_found = sorted(best_per_gene["gene_norm"].tolist())
    logger.info(f"  Genes: {', '.join(genes_found)}")

    # Flag rows with no gene name — report them
    if len(no_gene) > 0:
        logger.warning(
            f"  {len(no_gene):,} proteins have no gene name — "
            f"they will be handled in Step 2 (length clustering)"
        )

    return best_per_gene, no_gene


# ── Step 2: Length-based deduplication for ungened proteins ───────────────────

def deduplicate_by_length_cluster(no_gene_df: pd.DataFrame,
                                   existing_lengths: set) -> pd.DataFrame:
    """
    For proteins with no gene name, cluster by sequence length ± 50 aa
    (proteins differing by only a few aa are likely the same gene from
    different strains). Keep the longest per cluster.

    Also skip any length that's already represented in Step 1 results
    to avoid duplicating known genes.
    """
    if len(no_gene_df) == 0:
        return pd.DataFrame()

    logger.info("Step 2: Length-cluster deduplication for proteins with no gene name")

    df = no_gene_df.copy().sort_values("seq_length", ascending=False)
    clusters = []
    used_lengths = set()

    for _, row in df.iterrows():
        length = row["seq_length"]

        # Skip if a protein of similar length already chosen
        if any(abs(length - ul) <= 50 for ul in used_lengths):
            continue

        # Skip if this length is already covered by a gene-named protein
        if any(abs(length - el) <= 50 for el in existing_lengths):
            continue

        clusters.append(row)
        used_lengths.add(length)

    result = pd.DataFrame(clusters)
    logger.info(f"  Representative proteins from ungened set: {len(result)}")
    return result


# ── Step 3: Validation ────────────────────────────────────────────────────────

def validate_reference_proteome(df: pd.DataFrame) -> None:
    """
    Check the final reference set looks like a real SARS-CoV-2 proteome.
    Known length landmarks (Wuhan-Hu-1 UniProt UP000464024):
        Spike:        1,273 aa
        Nucleocapsid:   419 aa
        Membrane:       222 aa
        Envelope:        75 aa
        ORF1ab:       7,096 aa (or ~4,405 ORF1a + ~2,691 ORF1b)
    """
    console.rule("[bold green]Reference Proteome Validation[/bold green]")

    t = Table(title="SARS-CoV-2 Reference Proteins",
              header_style="bold cyan", show_lines=True)
    t.add_column("Gene",         style="white",       min_width=15)
    t.add_column("UniProt ID",   style="dim",          min_width=15)
    t.add_column("Protein name", style="white",        min_width=30)
    t.add_column("Length (aa)", style="bold yellow",   min_width=12)
    t.add_column("Check",        style="white",         min_width=10)

    # Expected lengths ± 10% for key proteins
    expected = {
        "S":      (1200, 1350),
        "N":      (380,  450),
        "M":      (200,  250),
        "E":      (60,   90),
        "ORF1ab": (6500, 7200),
    }

    for _, row in df.sort_values("seq_length", ascending=False).iterrows():
        gene = str(row.get("gene_norm", row.get("gene_name", "?")))
        lo, hi = expected.get(gene, (None, None))
        length = row["seq_length"]

        if lo is not None:
            ok = "✓" if lo <= length <= hi else f"⚠ expected {lo}–{hi}"
        else:
            ok = "—"

        t.add_row(
            gene,
            str(row.get("uniprot_id", "?")),
            str(row.get("protein_name", ""))[:30],
            f"{length:,}",
            ok,
        )

    console.print(t)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    console.rule("[bold cyan]COVID Proteome Fix: Reference Proteome Extraction[/bold cyan]")

    # Load original
    input_path = PROCESSED_DIR / "covid_proteins_clean.csv"
    if not input_path.exists():
        logger.error(f"  File not found: {input_path}")
        logger.error("  Run 01_clean_data_covid.py first.")
        sys.exit(1)

    df = pd.read_csv(input_path)
    logger.info(f"  Loaded covid_proteins_clean.csv: {len(df):,} rows")
    logger.info(f"  Columns: {list(df.columns)}")
    logger.info(f"  Length range: {df['seq_length'].min()}–{df['seq_length'].max()} aa")
    logger.info(f"  Unique gene names (raw): {df['gene_name'].nunique()}")

    # Show gene name distribution before fix
    logger.info("  Top gene names in raw file:")
    top_genes = df["gene_name"].value_counts().head(20)
    for gene, count in top_genes.items():
        logger.info(f"    '{gene}': {count:,}")

    print()

    # Step 1: gene-level dedup
    gene_deduped, no_gene = deduplicate_by_gene(df)

    # Step 2: handle proteins with no gene name
    existing_lengths = set(gene_deduped["seq_length"].tolist())
    ungened_deduped  = deduplicate_by_length_cluster(no_gene, existing_lengths)

    # Combine
    if len(ungened_deduped) > 0:
        final = pd.concat([gene_deduped, ungened_deduped], ignore_index=True)
    else:
        final = gene_deduped.copy()

    # Clean up helper column
    final = final.drop(columns=["gene_norm"], errors="ignore")
    final = final.reset_index(drop=True)

    logger.info(f"\n  BEFORE: {len(df):,} proteins")
    logger.info(f"  AFTER:  {len(final):,} proteins")
    logger.info(
        f"  Reduction: {len(df) - len(final):,} strain-variant duplicates removed"
    )

    # Validate
    validate_reference_proteome(final)

    # ── Decision point ────────────────────────────────────────────────────────
    print()
    if len(final) > 60:
        logger.warning(
            f"  Result has {len(final)} proteins — still more than expected (~29)."
        )
        logger.warning(
            "  This means many proteins in your FASTA have no gene_name annotation."
        )
        logger.warning(
            "  RECOMMENDATION: Re-download the proteome from UniProt using "
            "proteome ID UP000464024 (the canonical reference) instead of a "
            "taxon-level dump. See instructions below."
        )
        print()
        console.print("[bold red]ACTION REQUIRED:[/bold red]")
        console.print(
            "  Download the correct file from:\n"
            "  https://www.uniprot.org/proteomes/UP000464024\n"
            "  → Click 'Download' → Format: FASTA → Reviewed (Swiss-Prot) only\n"
            "  → Save as: data/raw/covid_proteome_reference.fasta\n"
            "  Then re-run 01_clean_data_covid.py pointing to this file.\n"
        )
    elif len(final) < 20:
        logger.warning(
            f"  Result has only {len(final)} proteins — fewer than expected (~29)."
        )
        logger.warning(
            "  Some canonical genes may be missing gene_name annotations "
            "in your FASTA headers. Check the protein_name column above."
        )
    else:
        logger.info(
            f"  Result looks correct ({len(final)} proteins ≈ 29 canonical genes)."
        )
        console.print("\n[bold green]Proteome fix successful![/bold green]")

    # Save regardless — even partial fix is better than 5,699
    output_path = PROCESSED_DIR / "covid_proteins_reference.csv"
    final.to_csv(output_path, index=False)
    logger.info(f"  Saved: {output_path.relative_to(PROJECT_ROOT)}")
    logger.info(
        "  Script 3 will use covid_proteins_reference.csv instead of "
        "covid_proteins_clean.csv"
    )

    print()
    console.print(
        "[bold cyan]Next step:[/bold cyan] "
        "uv run python scripts/03_feature_engineering_covid.py\n"
        "(script 3 has been updated to use covid_proteins_reference.csv)\n"
    )


if __name__ == "__main__":
    main()
