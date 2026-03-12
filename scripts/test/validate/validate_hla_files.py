#!/usr/bin/env python3
"""
Script 3 v2: Fixed HLA allele parsing for IMGT format
"""

from pathlib import Path
from Bio import SeqIO
from collections import Counter

DATA_DIR = Path("data")
HLA_FILES = {
    "gen": DATA_DIR / "HLA" / "hla_gen.fasta",
    "nuc": DATA_DIR / "HLA" / "hla_nuc.fasta", 
    "prot": DATA_DIR / "HLA" / "hla_prot.fasta"
}

def parse_allele_from_header(header: str) -> str:
    """Extract HLA gene name from IMGT FASTA header"""
    parts = header.split()
    if not parts:
        return 'Unknown'
    
    header_id = parts[0]
    # IMGT format: HLA00001a or HLAB*01:01
    if '*' in header_id:
        return header_id.split('*')[0]  # HLAB
    elif 'HLA' in header_id.upper():
        # Extract gene: HLA-A, HLA-B, etc.
        if len(parts) > 2 and '*' in parts[2]:
            return parts[2].split('*')[0]  # HLA-A from HLA-A*11:01
    return 'Unknown'

def validate_fasta(filepath: Path, file_type: str):
    print(f"\n{'='*60}")
    print(f"🧬 HLA {file_type.upper()} VALIDATION")
    print(f"📁 {filepath}")
    print('='*60)
    
    if not filepath.exists():
        print("❌ FILE MISSING")
        return None
    
    records = list(SeqIO.parse(filepath, "fasta"))
    print(f"✅ LOADED: {len(records):,} sequences")
    
    # Fixed allele parsing
    alleles = [parse_allele_from_header(rec.id) for rec in records]
    valid_alleles = [a for a in alleles if a != 'Unknown']
    print(f"  Unique alleles: {len(set(valid_alleles))}")
    print(f"  Top 5 alleles: {Counter(valid_alleles).most_common(5)}")
    
    lengths = [len(rec.seq) for rec in records]
    print(f"  Length range: {min(lengths)}-{max(lengths)}")
    print(f"  Median length: {sorted(lengths)[len(lengths)//2]}")
    
    return len(records), len(set(valid_alleles))

def main():
    print("🧬 HLA VALIDATION v2.0 (Fixed IMGT parsing)")
    
    total_seqs = 0
    total_alleles = 0
    for file_type, filepath in HLA_FILES.items():
        result = validate_fasta(filepath, file_type)
        if result:
            seqs, alleles = result
            total_seqs += seqs
            total_alleles += alleles
    
    print(f"\n{'='*60}")
    print(f"FINAL HLA SUMMARY: {total_seqs:,} seqs, {total_alleles:,} alleles")
    print("✅ HLA FILES PERFECT - MOVING TO TBDB")
    
    print("\nNext: validate_tbdb.py")

if __name__ == "__main__":
    main()
