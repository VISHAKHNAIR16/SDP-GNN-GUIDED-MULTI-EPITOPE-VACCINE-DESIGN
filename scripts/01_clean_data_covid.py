"""
01_clean_data_covid.py
======================
Phase 1 (COVID validation): Data Cleaning
Same pipeline as TB but adapted for SARS-CoV-2 datasets.

Column name differences handled:
    IEDB:  'Epitope - Name'           → epitope_seq
           'Epitope - Source Molecule' → source_molecule
           'Epitope - Source Organism' → source_organism
    VDJdb: 'CDR3'                     → cdr3
           'Epitope'                  → epitope
           'Species'                  → species
           'MHC A'                    → mhc_a

Run from project root:
    uv run python scripts/01_clean_data_covid.py
"""

import sys
import re
import zipfile
import io
from pathlib import Path

import pandas as pd
from loguru import logger
from Bio import SeqIO

# ── Setup ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
RAW_DIR       = PROJECT_ROOT / "data" / "raw"
OUT_DIR       = PROJECT_ROOT / "data" / "processed_covid"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

VALID_AA = set("ACDEFGHIKLMNPQRSTVWYBZXU")

def is_valid_peptide(seq, min_len=8, max_len=25):
    if not isinstance(seq, str): return False
    seq = seq.strip().upper()
    if len(seq) < min_len or len(seq) > max_len: return False
    if not all(c in VALID_AA for c in seq): return False
    return True

# ── Step 1: Clean IEDB positive epitopes ─────────────────────────────────────

def clean_iedb_positive():
    logger.info("Cleaning IEDB COVID positive epitopes...")

    path = RAW_DIR / "covid_positive.csv"

    # IEDB downloads as a zip — handle both zip and plain csv
    try:
        with zipfile.ZipFile(str(path)) as z:
            fname = z.namelist()[0]
            with z.open(fname) as f:
                df = pd.read_csv(f, low_memory=False)
    except zipfile.BadZipFile:
        df = pd.read_csv(str(path), low_memory=False)

    logger.info(f"  Raw rows: {len(df):,}")

    # Rename columns to match our standard names
    df = df.rename(columns={
        "Epitope - Name":            "epitope_seq",
        "Epitope - Source Molecule": "source_molecule",
        "Epitope - Source Organism": "source_organism",
        "Epitope - Object Type":     "object_type",
    })

    # Keep only linear peptides
    df = df[df["object_type"].str.lower().str.contains("linear", na=False)]
    logger.info(f"  After linear peptide filter: {len(df):,}")

    # Keep only SARS-CoV-2 source
    covid_mask = df["source_organism"].str.contains(
        "SARS|coronavirus 2|COVID", case=False, na=False
    )
    df = df[covid_mask]
    logger.info(f"  After SARS-CoV-2 filter: {len(df):,}")

    # Clean sequence
    df["epitope_seq"] = df["epitope_seq"].astype(str).str.strip().str.upper()

    # Validate
    valid_mask = df["epitope_seq"].apply(is_valid_peptide)
    df = df[valid_mask]
    logger.info(f"  After validation (8-25aa, valid AA): {len(df):,}")

    # Deduplicate
    df = df.drop_duplicates(subset=["epitope_seq"])
    logger.info(f"  After deduplication: {len(df):,}")

    # Add label
    df["label"] = 1

    # Keep only needed columns
    out = df[["epitope_seq", "source_molecule", "source_organism", "label"]].reset_index(drop=True)

    out.to_csv(OUT_DIR / "iedb_positive_covid.csv", index=False)
    logger.info(f"  Saved iedb_positive_covid.csv — {len(out):,} unique positive epitopes")
    return out

# ── Step 2: Clean IEDB negative epitopes ─────────────────────────────────────

def clean_iedb_negative():
    logger.info("Cleaning IEDB COVID negative epitopes...")

    path = RAW_DIR / "covid_negative.csv"

    try:
        with zipfile.ZipFile(str(path)) as z:
            fname = z.namelist()[0]
            with z.open(fname) as f:
                df = pd.read_csv(f, low_memory=False)
    except zipfile.BadZipFile:
        df = pd.read_csv(str(path), low_memory=False)

    logger.info(f"  Raw rows: {len(df):,}")

    df = df.rename(columns={
        "Epitope - Name":            "epitope_seq",
        "Epitope - Source Molecule": "source_molecule",
        "Epitope - Source Organism": "source_organism",
        "Epitope - Object Type":     "object_type",
    })

    # Linear peptides only
    df = df[df["object_type"].str.lower().str.contains("linear", na=False)]

    # SARS-CoV-2 only
    df = df[df["source_organism"].str.contains(
        "SARS|coronavirus 2|COVID", case=False, na=False
    )]

    df["epitope_seq"] = df["epitope_seq"].astype(str).str.strip().str.upper()
    df = df[df["epitope_seq"].apply(is_valid_peptide)]
    df = df.drop_duplicates(subset=["epitope_seq"])
    logger.info(f"  After cleaning + dedup: {len(df):,}")

    df["label"] = 0
    out = df[["epitope_seq", "source_molecule", "source_organism", "label"]].reset_index(drop=True)

    out.to_csv(OUT_DIR / "iedb_negative_covid.csv", index=False)
    logger.info(f"  Saved iedb_negative_covid.csv — {len(out):,} unique negative epitopes")
    return out

# ── Step 3: Clean VDJdb COVID TCR data ───────────────────────────────────────

def clean_vdjdb():
    logger.info("Cleaning VDJdb COVID TCR data...")

    path = RAW_DIR / "covid_vdjdb.tsv"
    df = pd.read_csv(str(path), sep="\t", low_memory=False)
    logger.info(f"  Raw rows: {len(df):,}")

    # Rename to standard names
    df = df.rename(columns={
        "CDR3":            "cdr3",
        "Epitope":         "epitope",
        "Epitope species": "epitope_species",
        "Species":         "species",
        "MHC A":           "mhc_a",
        "MHC class":       "mhc_class",
        "Score":           "score",
        "Gene":            "gene",
        "V":               "v_gene",
        "J":               "j_gene",
    })

    # Keep SARS-CoV-2 only
    covid_mask = df["epitope_species"].str.contains(
        "SARS|COVID|coronavirus", case=False, na=False
    )
    df = df[covid_mask]
    logger.info(f"  After SARS-CoV-2 filter: {len(df):,}")

    # Keep human only — VDJdb uses 'HomoSapiens' (no space)
    human_mask = df["species"].str.lower().str.contains("homo|human", na=False)
    df = df[human_mask]
    logger.info(f"  After human filter: {len(df):,}")

    # Validate CDR3
    df["cdr3"] = df["cdr3"].astype(str).str.strip().str.upper()
    valid_cdr3 = df["cdr3"].apply(
        lambda s: isinstance(s, str) and 5 <= len(s) <= 50
                  and all(c in VALID_AA for c in s)
    )
    df = df[valid_cdr3]
    logger.info(f"  After CDR3 validation: {len(df):,}")

    # Validate epitope sequences
    df["epitope"] = df["epitope"].astype(str).str.strip().str.upper()
    df = df[df["epitope"].apply(lambda s: is_valid_peptide(s, min_len=5, max_len=30))]

    # Deduplicate on (cdr3, epitope) pair
    df = df.drop_duplicates(subset=["cdr3", "epitope"])
    logger.info(f"  After deduplication on (CDR3, epitope): {len(df):,}")
    logger.info(f"  Unique CDR3 sequences: {df['cdr3'].nunique():,}")
    logger.info(f"  Unique COVID epitopes with TCR data: {df['epitope'].nunique():,}")

    df.to_csv(OUT_DIR / "vdjdb_covid_clean.tsv", sep="\t", index=False)
    logger.info("  Saved vdjdb_covid_clean.tsv")
    return df

# ── Step 4: Clean COVID proteome ──────────────────────────────────────────────

def clean_proteome():
    logger.info("Cleaning SARS-CoV-2 proteome...")

    path = RAW_DIR / "covid_proteome.fasta"
    records = []

    for rec in SeqIO.parse(str(path), "fasta"):
        seq = str(rec.seq).upper()

        # Skip if too short
        if len(seq) < 50:
            continue

        # Skip if too many ambiguous residues
        ambig = sum(1 for c in seq if c not in "ACDEFGHIKLMNPQRSTVWY")
        if ambig / len(seq) > 0.1:
            continue

        # Parse gene name from description
        gene_name = ""
        if "GN=" in rec.description:
            gene_name = rec.description.split("GN=")[1].split()[0]

        # Parse protein name — text between first space and 'OS='
        prot_name = ""
        desc = rec.description
        if " " in desc:
            rest = desc[desc.index(" ")+1:]
            if "OS=" in rest:
                prot_name = rest.split("OS=")[0].strip()
            else:
                prot_name = rest[:80]

        records.append({
            "uniprot_id":   rec.id,
            "protein_name": prot_name[:100],
            "gene_name":    gene_name,
            "sequence":     seq,
            "seq_length":   len(seq),
        })

    df = pd.DataFrame(records)
    logger.info(f"  Raw proteins: {len(df):,}")

    # Deduplicate by sequence
    df = df.drop_duplicates(subset=["sequence"])
    logger.info(f"  After dedup by sequence: {len(df):,}")

    df.to_csv(OUT_DIR / "covid_proteins_clean.csv", index=False)
    logger.info(f"  Saved covid_proteins_clean.csv")
    logger.info(f"  Gene names found: {df['gene_name'].nunique():,} unique genes")
    logger.info(f"  Avg protein length: {df['seq_length'].mean():.0f} aa")
    return df

# ── Step 5: Summary ───────────────────────────────────────────────────────────

def print_summary(pos, neg, vdj, prot):
    print("\n" + "="*55)
    print("  COVID DATA CLEANING SUMMARY")
    print("="*55)
    print(f"  Positive epitopes (immunogenic):  {len(pos):>6,}")
    print(f"  Negative epitopes (non-immunog.): {len(neg):>6,}")
    total = len(pos) + len(neg)
    ratio = len(neg) / len(pos) if len(pos) > 0 else 0
    print(f"  Total epitopes:                   {total:>6,}")
    print(f"  Class imbalance ratio:            {ratio:.1f}:1")
    print(f"  VDJdb TCR-epitope pairs:          {len(vdj):>6,}")
    print(f"  Unique COVID TCR CDR3s:           {vdj['cdr3'].nunique():>6,}")
    print(f"  Unique epitopes with TCR data:    {vdj['epitope'].nunique():>6,}")
    print(f"  COVID proteome proteins:          {len(prot):>6,}")
    print("="*55)

    # Overlap — epitopes in both IEDB positive AND VDJdb
    vjdb_eps  = set(vdj["epitope"].str.upper())
    iedb_pos  = set(pos["epitope_seq"].str.upper())
    overlap   = vjdb_eps & iedb_pos
    print(f"\n  Gold standard overlap")
    print(f"  (IEDB positive + VDJdb confirmed): {len(overlap):,}")
    if overlap:
        print("  Overlapping epitopes:")
        for ep in sorted(overlap)[:10]:
            print(f"    {ep}")
    print()
    print(f"  Output folder: {OUT_DIR.relative_to(PROJECT_ROOT)}")
    print("\n  Next step: uv run python scripts/02_eda_covid.py")
    print("="*55)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*55)
    print("  Phase 1 (COVID): Data Cleaning")
    print("="*55 + "\n")

    pos  = clean_iedb_positive()
    neg  = clean_iedb_negative()
    vdj  = clean_vdjdb()
    prot = clean_proteome()

    print_summary(pos, neg, vdj, prot)

if __name__ == "__main__":
    main()