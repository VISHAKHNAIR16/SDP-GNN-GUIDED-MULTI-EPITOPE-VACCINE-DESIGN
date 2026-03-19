"""
01_clean_data.py  (v3 — Final)
================================
Phase 1: Data Cleaning for GNN-Guided Multi-Epitope Vaccine Design

All fixes applied:
    FIX 1 - IEDB sequence column:
            'peptide_seq' = "Linear peptide" (epitope TYPE, not the sequence).
            Real sequence is in 'Epitope|Name' -> becomes 'epitope_name' after
            column standardization. Script now auto-detects by checking which
            column actually contains amino-acid-like values.

    FIX 2 - IEDB is the FULL database (all organisms).
            Must filter to Mycobacterium tuberculosis using
            'Epitope|Source Organism' column before deduplication.

    FIX 3 - VDJdb species = "HomoSapiens" (CamelCase, no space).
            TB filter uses 'epitope_species' column.

    FIX 4 - HLA categorization reads description field for gene name
            (IPD-IMGT IDs are numeric, gene is in allele notation A*01:01).

Run from project root:
    uv run python scripts/01_clean_data.py
"""

import re
import sys
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from loguru import logger
from rich.console import Console
from rich.table import Table

# ── Setup ─────────────────────────────────────────────────────────────────────

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

out_dir = PROJECT_ROOT / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "figures").mkdir(parents=True, exist_ok=True)
(out_dir / "models").mkdir(parents=True, exist_ok=True)
logger.add(out_dir / "cleaning.log", rotation="1 MB")

# ── Constants ─────────────────────────────────────────────────────────────────

VALID_AA       = set("ACDEFGHIKLMNPQRSTVWYBZXU")
EPITOPE_MIN    = 8
EPITOPE_MAX    = 25

TB_ORG_PATTERN = r"mycobacterium tuberculosis|m\. tuberculosis|mtb"

# ── Helpers ───────────────────────────────────────────────────────────────────

def is_valid_peptide(seq: str) -> bool:
    if not isinstance(seq, str):
        return False
    s = seq.upper().strip()
    return EPITOPE_MIN <= len(s) <= EPITOPE_MAX and all(c in VALID_AA for c in s)


def is_valid_protein(seq: str, min_len: int = 50) -> bool:
    if not isinstance(seq, str):
        return False
    s = seq.upper().strip().rstrip("*")
    return len(s) >= min_len and all(c in VALID_AA for c in s)


def std_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize DataFrame column names: lowercase, spaces→underscore, strip special chars."""
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\w]", "_", regex=True)
    )
    return df


def rich_table(title: str, stats: dict) -> None:
    t = Table(title=title, show_header=True, header_style="bold cyan")
    t.add_column("Metric", style="white")
    t.add_column("Value",  style="bold yellow")
    for k, v in stats.items():
        t.add_row(str(k), str(v))
    console.print(t)


def find_peptide_column(df: pd.DataFrame) -> str | None:
    """
    Auto-detect which column contains amino acid sequences.

    Why needed: IEDB's 'peptide_seq' column actually stores the epitope
    object type ('Linear peptide'), not the sequence itself. The real
    sequence is in 'epitope_name'. We verify by checking the values.
    """
    # Check these candidates in priority order
    candidates = [
        "epitope_name", "name", "peptide_seq",
        "description", "epitope_description", "sequence", "peptide",
    ]
    for col in candidates:
        if col not in df.columns:
            continue
        sample = df[col].dropna().astype(str).head(30)
        peptide_like = sum(
            1 for v in sample
            if EPITOPE_MIN <= len(v.strip()) <= EPITOPE_MAX
            and all(c in VALID_AA for c in v.strip().upper())
            and v.strip().isalpha()
        )
        if peptide_like >= 5:
            logger.info(f"  Sequence column: '{col}' "
                        f"({peptide_like}/30 sample values look like peptides)")
            return col
        else:
            logger.info(f"  Skipping '{col}': only {peptide_like}/30 values are peptide-like "
                        f"(sample: '{str(df[col].iloc[0])[:40]}')")
    return None


# ── Cleaner 1: IEDB ───────────────────────────────────────────────────────────

def clean_iedb_epitopes(
    input_path: Path,
    output_filename: str,
    label: int,
) -> pd.DataFrame:
    """
    Clean IEDB T-cell epitope export.

    IEDB export structure (confirmed):
      - 'peptide_seq'    -> "Linear peptide" (object type, NOT the sequence)
      - 'Epitope|Name'   -> actual amino acid sequence e.g. "KLGGALQAK"
      - One row per assay experiment, not per epitope
      - Contains ALL organisms — must filter to TB only

    Steps:
      1. Auto-detect real sequence column
      2. Filter to Mycobacterium tuberculosis entries
      3. Validate sequence characters and length
      4. Deduplicate: one row per unique sequence (keep first)
      5. Save with useful metadata columns
    """
    logger.info(f"Cleaning IEDB: {input_path.name}")
    df = pd.read_csv(input_path, low_memory=False)
    raw_count = len(df)
    logger.info(f"  Loaded {raw_count} rows (all organisms, one row per assay)")

    df = std_cols(df)

    # ── 1. Find the actual sequence column ──
    seq_col = find_peptide_column(df)
    if seq_col is None:
        logger.error("  Cannot find peptide sequence column!")
        logger.error(f"  All columns: {list(df.columns)}")
        sys.exit(1)
    df = df.rename(columns={seq_col: "epitope_seq"})

    # ── 2. Filter to Mycobacterium tuberculosis only ──
    # 'Epitope|Source Organism' -> 'epitope_source_organism' after std_cols
    org_col = next(
        (c for c in df.columns if "source_organism" in c),
        next((c for c in df.columns if "organism" in c), None)
    )
    if org_col:
        before = len(df)
        tb_mask = df[org_col].astype(str).str.lower().str.contains(TB_ORG_PATTERN, na=False)
        df = df[tb_mask].copy()
        logger.info(f"  TB filter ('{org_col}'): {before} -> {len(df)} rows")
        if len(df) == 0:
            logger.error("  TB filter returned 0 rows. Top organisms in raw file:")
            raw = pd.read_csv(input_path, low_memory=False)
            raw = std_cols(raw)
            for org, cnt in raw[org_col].value_counts().head(10).items():
                logger.error(f"    '{org}' -> {cnt}")
    else:
        logger.warning("  No source organism column found — NOT filtering by organism!")
        logger.warning(f"  Columns with 'org': {[c for c in df.columns if 'org' in c]}")

    # ── 3. Clean and validate sequences ──
    df["epitope_seq"] = (
        df["epitope_seq"].astype(str).str.upper().str.strip()
        .str.replace(r"\s+", "", regex=True)
    )
    before = len(df)
    df = df[df["epitope_seq"].apply(is_valid_peptide)].copy()
    logger.info(f"  Removed {before - len(df)} invalid/out-of-range sequences")

    # ── 4. Deduplicate ──
    unique_count = df["epitope_seq"].nunique()
    total_assays  = len(df)
    logger.info(f"  {unique_count} unique TB epitopes across {total_assays} assay rows "
                f"(avg {total_assays / max(unique_count, 1):.1f}x repetition)")
    df = df.drop_duplicates(subset=["epitope_seq"], keep="first")
    logger.info(f"  After dedup: {len(df)} unique epitopes")

    # ── 5. Add label and length ──
    df["label"]      = label
    df["seq_length"] = df["epitope_seq"].str.len()

    # ── 6. Keep useful columns ──
    keep = ["epitope_seq", "seq_length", "label"]
    for col in [org_col, "epitope_source_molecule",
                "epitope_starting_position", "epitope_ending_position",
                "epitope_iri", "epitope_species"]:
        if col and col in df.columns and col not in keep:
            keep.append(col)

    df_clean = df[keep].reset_index(drop=True)
    out_path = PROCESSED_DIR / output_filename
    df_clean.to_csv(out_path, index=False)

    rich_table(f"IEDB {'Positive' if label == 1 else 'Negative'} Epitopes", {
        "Raw rows (all organisms)":  raw_count,
        "After TB filter":           total_assays,
        "Unique TB epitopes":        len(df_clean),
        "Avg assay repetitions":     f"{total_assays / max(len(df_clean), 1):.1f}x",
        "Label":                     "Positive (immunogenic)" if label == 1 else "Negative",
        "Length range":              f"{df_clean['seq_length'].min()}-{df_clean['seq_length'].max()} aa",
        "Saved to":                  out_path.relative_to(PROJECT_ROOT),
    })
    return df_clean


# ── Cleaner 2: VDJdb ──────────────────────────────────────────────────────────

def clean_vjdb(input_path: Path) -> pd.DataFrame:
    """
    Clean VDJdb TCR-epitope binding data (TB + human only).

    Confirmed columns:
        species, epitope_species, cdr3, epitope, score, mhc_a, mhc_b, gene

    Filters:
        species       -> "HomoSapiens" (normalize: remove spaces, lowercase)
        epitope_species -> "Mycobacterium tuberculosis"
        score         -> >= 1  (VDJdb quality metric: 0=unverified, 1=medium, 2=high)
        cdr3          -> valid amino acid string, 5–40 aa
    """
    logger.info(f"Cleaning VDJdb: {input_path.name}")
    df = pd.read_csv(input_path, sep="\t", low_memory=False)
    raw_count = len(df)
    df = std_cols(df)
    logger.info(f"  Loaded {raw_count} rows | columns: {list(df.columns)}")

    # Human filter
    before = len(df)
    sp_norm = df["species"].astype(str).str.replace(r"\s+", "", regex=True).str.lower()
    df = df[sp_norm.str.contains("homosapiens|human", na=False)].copy()
    logger.info(f"  Human filter: {before} -> {len(df)}")

    # TB filter
    before = len(df)
    es_norm = df["epitope_species"].astype(str).str.replace(r"\s+", "", regex=True).str.lower()
    df = df[es_norm.str.contains("mycobacteriumtuberculosis|tuberculosis|mtb", na=False)].copy()
    logger.info(f"  TB filter: {before} -> {len(df)}")

    # Confidence score >= 1
    before = len(df)
    df = df[pd.to_numeric(df["score"], errors="coerce").fillna(0) >= 1].copy()
    logger.info(f"  Score >= 1: {before} -> {len(df)}")

    # CDR3 validation
    df["cdr3"] = df["cdr3"].astype(str).str.upper().str.strip()
    before = len(df)
    valid = df["cdr3"].apply(
        lambda s: all(c in VALID_AA for c in s) and 5 <= len(s) <= 40
    )
    df = df[valid].copy()
    logger.info(f"  CDR3 validation: {before} -> {len(df)}")

    # Deduplicate
    before = len(df)
    df = df.drop_duplicates(subset=["cdr3", "epitope"])
    logger.info(f"  Dedup (CDR3+epitope): removed {before - len(df)}")

    df = df.reset_index(drop=True)
    out_path = PROCESSED_DIR / "vjdb_tb_human_clean.tsv"
    df.to_csv(out_path, sep="\t", index=False)

    rich_table("VDJdb TCR Data", {
        "Raw rows":       raw_count,
        "After cleaning": len(df),
        "Unique epitopes in TCR data": df["epitope"].nunique(),
        "Unique CDR3s":   df["cdr3"].nunique(),
        "Saved to":       out_path.relative_to(PROJECT_ROOT),
    })
    return df


# ── Cleaner 3: HLA FASTA ─────────────────────────────────────────────────────

def clean_hla_fasta(input_path: Path) -> list:
    """
    Clean HLA protein FASTA from IPD-IMGT.

    ID format:   HLA:HLA00001
    Description: HLA:HLA00001 A*01:01:01:01 365 bp

    Gene extracted from allele notation GENE*XX:XX in description.
    """
    logger.info(f"Cleaning HLA FASTA: {input_path.name}")
    records   = list(SeqIO.parse(str(input_path), "fasta"))
    raw_count = len(records)
    for i, r in enumerate(records[:3]):
        logger.info(f"  Sample[{i}]: '{r.description[:80]}'")

    clean_records = []
    seen_ids = set()
    counts = {"A": 0, "B": 0, "C": 0, "DR": 0, "DQ/DP": 0, "Other": 0, "Removed": 0}

    for rec in records:
        seq = str(rec.seq).upper()
        if not is_valid_protein(seq):
            counts["Removed"] += 1; continue
        if seq.count("X") / len(seq) > 0.1:
            counts["Removed"] += 1; continue
        if rec.id in seen_ids:
            counts["Removed"] += 1; continue
        seen_ids.add(rec.id)

        m    = re.search(r"\b([A-Z0-9]+)\*\d+:\d+", rec.description.upper())
        gene = m.group(1) if m else ""
        if   gene == "A":                                  counts["A"]     += 1
        elif gene == "B":                                  counts["B"]     += 1
        elif gene == "C":                                  counts["C"]     += 1
        elif gene in ("DRA","DRB1","DRB3","DRB4","DRB5"): counts["DR"]    += 1
        elif gene in ("DQA1","DQB1","DPA1","DPB1"):        counts["DQ/DP"] += 1
        else:                                              counts["Other"] += 1

        rec.description = rec.description.strip()
        clean_records.append(rec)

    out_path = PROCESSED_DIR / "hla_prot_clean.fasta"
    SeqIO.write(clean_records, str(out_path), "fasta")

    rich_table("HLA Protein FASTA", {
        "Raw sequences":      raw_count,
        "After cleaning":     len(clean_records),
        "HLA-A":              counts["A"],
        "HLA-B":              counts["B"],
        "HLA-C":              counts["C"],
        "HLA-DR":             counts["DR"],
        "HLA-DQ/DP":          counts["DQ/DP"],
        "Other/Non-classical":counts["Other"],
        "Removed":            counts["Removed"],
        "Saved to":           out_path.relative_to(PROJECT_ROOT),
    })
    return clean_records


# ── Cleaner 4: TB Proteome ────────────────────────────────────────────────────

def clean_tb_proteome(input_path: Path) -> tuple:
    """
    Clean M. tuberculosis H37Rv proteome FASTA.
    Extracts UniProt IDs and gene names, saves FASTA + metadata CSV.
    """
    logger.info(f"Cleaning TB proteome: {input_path.name}")
    records   = list(SeqIO.parse(str(input_path), "fasta"))
    raw_count = len(records)
    if records:
        logger.info(f"  Sample: '{records[0].description[:100]}'")

    clean_records, protein_info, seen_ids = [], [], set()

    for rec in records:
        seq = str(rec.seq).upper().rstrip("*")
        if not is_valid_protein(seq):
            continue

        uid = rec.id
        if "|" in uid:
            parts = uid.split("|")
            uid = parts[1] if len(parts) >= 2 else uid
        if uid in seen_ids:
            continue
        seen_ids.add(uid)

        gene = ""
        m = re.search(r"GN=(\S+)", rec.description)
        if m:
            gene = m.group(1)

        pname = rec.description
        m2 = re.search(r"^(.*?)\s+OS=", pname)
        if m2:
            raw = m2.group(1)
            parts = raw.split(" ", 1)
            pname = parts[1] if len(parts) > 1 else raw

        clean_records.append(SeqRecord(rec.seq, id=uid, name=gene or uid, description=pname))
        protein_info.append({"uniprot_id": uid, "gene_name": gene,
                              "protein_name": pname[:80], "seq_length": len(seq)})

    df_prot = pd.DataFrame(protein_info)
    SeqIO.write(clean_records, str(PROCESSED_DIR / "tb_proteome_clean.fasta"), "fasta")
    df_prot.to_csv(PROCESSED_DIR / "tb_proteome_metadata.csv", index=False)

    rich_table("M. tuberculosis H37Rv Proteome", {
        "Raw proteins":    raw_count,
        "After cleaning":  len(clean_records),
        "Avg length":      f"{df_prot['seq_length'].mean():.0f} aa",
        "Min / Max":       f"{df_prot['seq_length'].min()} / {df_prot['seq_length'].max()} aa",
        "With gene names": df_prot['gene_name'].ne('').sum(),
        "Saved FASTA":     "data/processed/tb_proteome_clean.fasta",
        "Saved metadata":  "data/processed/tb_proteome_metadata.csv",
    })
    return clean_records, df_prot


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    console.rule("[bold cyan]Phase 1: Data Cleaning (v3 — Final)[/bold cyan]")
    console.print("\n[bold]Project:[/bold] GNN-Guided Multi-Epitope Vaccine Design\n")

    console.rule("[yellow]1/4  IEDB Positive Epitopes[/yellow]")
    pos_path = DATA_DIR / "IEDB" / "epitope_table_positive_clean.csv"
    if not pos_path.exists():
        pos_path = DATA_DIR / "IEDB" / "epitope_table_positive.csv"
    df_pos = clean_iedb_epitopes(pos_path, "iedb_positive_clean.csv", label=1)

    console.rule("[yellow]2/4  IEDB Negative Epitopes[/yellow]")
    neg_path = DATA_DIR / "OPTIONAL" / "epitope_table_negative_clean.csv"
    if not neg_path.exists():
        neg_path = DATA_DIR / "OPTIONAL" / "epitope_table_negative.csv"
    df_neg = clean_iedb_epitopes(neg_path, "iedb_negative_clean.csv", label=0)

    console.rule("[yellow]3/4  VDJdb TCR-Epitope Data[/yellow]")
    df_vjdb = clean_vjdb(DATA_DIR / "VJDB" / "vjdb.tsv")

    console.rule("[yellow]4/4  HLA + TB Proteome[/yellow]")
    hla_recs        = clean_hla_fasta(DATA_DIR / "HLA" / "hla_prot.fasta")
    tb_recs, df_tb  = clean_tb_proteome(DATA_DIR / "TBDB" / "tb1584.fasta")

    # ── Summary ──
    console.rule("[bold green]Cleaning Complete[/bold green]")
    t = Table(title="Processed Data Summary", header_style="bold green")
    t.add_column("Dataset",  style="white")
    t.add_column("Records",  style="bold yellow", justify="right")
    t.add_column("File",     style="dim")
    t.add_row("IEDB Positive (TB)",  str(len(df_pos)),   "iedb_positive_clean.csv")
    t.add_row("IEDB Negative (TB)",  str(len(df_neg)),   "iedb_negative_clean.csv")
    t.add_row("VDJdb TCR (TB+Human)",str(len(df_vjdb)),  "vjdb_tb_human_clean.tsv")
    t.add_row("HLA sequences",       str(len(hla_recs)), "hla_prot_clean.fasta")
    t.add_row("TB proteins",         str(len(tb_recs)),  "tb_proteome_clean.fasta")
    t.add_row("TB protein metadata", str(len(df_tb)),    "tb_proteome_metadata.csv")
    console.print(t)

    # ── Health check ──
    issues = []
    if len(df_pos) < 100:
        issues.append(f"IEDB positive: only {len(df_pos)} epitopes (expected 200+)")
    if len(df_neg) < 50:
        issues.append(f"IEDB negative: only {len(df_neg)} epitopes (expected 100+)")
    if len(df_vjdb) == 0:
        issues.append("VDJdb: 0 rows — check species/epitope_species filters")
    if len(hla_recs) < 1000:
        issues.append(f"HLA: only {len(hla_recs)} sequences")

    console.print()
    if issues:
        console.print("[bold red]Health Check — Issues:[/bold red]")
        for i in issues:
            console.print(f"  [red]• {i}[/red]")
    else:
        console.print("[bold green]Health check passed![/bold green]")

    console.print("\n[bold cyan]Next:[/bold cyan] uv run python scripts/02_eda.py\n")


if __name__ == "__main__":
    main()