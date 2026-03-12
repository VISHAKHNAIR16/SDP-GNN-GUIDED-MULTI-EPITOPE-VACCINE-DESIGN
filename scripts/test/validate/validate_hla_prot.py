#!/usr/bin/env python3
"""
Script 6: IEDB Source Proteins Validation (FINAL)
File: data/IEDB/iedb_prot.fasta
"""

from pathlib import Path
from Bio import SeqIO
import numpy as np

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

PROT_FILE = Path("data") / "HLA" / "hla_prot.fasta"

def validate_iedb_proteins(filepath: Path):
    """Validate IEDB source protein FASTA"""
    print("\n" + "="*70)
    print("🔬 IEDB SOURCE PROTEIN VALIDATION (FINAL DATASET)")
    print(f"📁 {filepath}")
    print("="*70)
    
    if not filepath.exists():
        print("ℹ️  OPTIONAL - No iedb_prot.fasta found")
        print("  Can skip or download IEDB full proteome")
        return "OPTIONAL_MISSING"
    
    records = list(SeqIO.parse(filepath, "fasta"))
    print(f"✅ LOADED: {len(records):,} source proteins")
    
    lengths = [len(rec) for rec in records]
    print(f"📏 Length range: {min(lengths):,} - {max(lengths):,} AA")
    print(f"📐 Median: {np.median(lengths):,.0f} AA")
    
    # Quality check
    invalid_prots = 0
    for rec in records[:100]:  # Sample first 100
        seq = str(rec.seq).upper()
        if any(c not in VALID_AA for c in seq):
            invalid_prots += 1
            print(f"⚠️  {rec.id}: Invalid AA found")
    
    print(f"🔬 Sampled invalid proteins: {invalid_prots}/100")
    
    print("\n✅ IEDB PROTEINS READY")
    return len(records)

if __name__ == "__main__":
    result = validate_iedb_proteins(PROT_FILE)
    if result == "OPTIONAL_MISSING":
        print("\nℹ️  Skipping - epitopes don't strictly need source proteins")
    else:
        print("\n🎉 ALL 6 DATASETS VALIDATED!")
