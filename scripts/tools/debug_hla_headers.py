#!/usr/bin/env python3
"""
Debug: Print first 5 headers from each HLA FASTA
"""

from pathlib import Path
from Bio import SeqIO

HLA_FILES = {
    "gen": "data/HLA/hla_gen.fasta",
    "nuc": "data/HLA/hla_nuc.fasta", 
    "prot": "data/HLA/hla_prot.fasta"
}

for name, filepath in HLA_FILES.items():
    print(f"\n=== {name.upper()} FILE ===")
    path = Path(filepath)
    if path.exists():
        with open(path) as f:
            lines = [next(f) for _ in range(10) if f.readline().startswith(">")]
            for line in lines[:5]:
                print(repr(line.strip()))  # Shows exact format
    else:
        print("❌ MISSING")
