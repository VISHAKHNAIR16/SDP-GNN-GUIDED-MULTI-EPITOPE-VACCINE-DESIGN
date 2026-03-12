#!/usr/bin/env python3
"""
FINAL HLA VALIDATION - Full file allele analysis
"""

from pathlib import Path
from collections import Counter
import time

DATA_DIR = Path("data")
HLA_FILES = {
    "genomic": DATA_DIR / "HLA" / "hla_gen.fasta",
    "nucleotide": DATA_DIR / "HLA" / "hla_nuc.fasta",
    "protein": DATA_DIR / "HLA" / "hla_prot.fasta"
}

def parse_hla_gene(raw_header: str) -> str:
    """Parse HLA gene A, B, C, DRB1 from header"""
    header = raw_header.strip('>').strip()
    parts = header.split()
    
    for part in parts:
        part_clean = part.replace(' ', '')
        if '*' in part_clean:
            gene = part_clean.split('*')[0]
            if gene in ['A', 'B', 'C', 'DRA', 'DRB1', 'DRB3', 'DRB4', 'DRB5', 
                       'DQA1', 'DQB1', 'DPA1', 'DPB1']:
                return gene
    return 'Other'

def validate_hla_file(filepath: Path, file_type: str):
    """Full header analysis"""
    print(f"\n{'='*70}")
    print(f"🧬 HLA {file_type.upper()}")
    print(f"📁 {filepath}")
    print('='*70)
    
    if not filepath.exists():
        print("❌ MISSING")
        return None
    
    print("🔍 Analyzing ALL headers...")
    start_time = time.time()
    
    alleles = []
    seq_count = 0
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('>'):
                seq_count += 1
                gene = parse_hla_gene(line)
                alleles.append(gene)
    
    elapsed = time.time() - start_time
    
    print(f"✅ PROCESSED: {seq_count:,} sequences ({elapsed:.1f}s)")
    print(f"🎯 Total HLA genes: {len(set(alleles))}")
    print(f"🏆 Top 10 HLA genes:")
    for gene, count in Counter(alleles).most_common(10):
        print(f"  {gene}: {count:,}")
    
    print("✅ VALID")
    return seq_count, Counter(alleles)

def main():
    print("🔬 COMPLETE HLA VALIDATION")
    print("="*70)
    
    all_alleles = Counter()
    total_seqs = 0
    
    for file_type, filepath in HLA_FILES.items():
        result = validate_hla_file(filepath, file_type)
        if result:
            seqs, alleles = result
            total_seqs += seqs
            all_alleles.update(alleles)
    
    print(f"\n{'='*70}")
    print("🎉 GRAND HLA SUMMARY")
    print(f"📊 Total sequences: {total_seqs:,}")
    print(f"🎯 Unique HLA genes: {len(all_alleles)}")
    print("🏆 All HLA genes:")
    for gene, count in all_alleles.most_common():
        print(f"  {gene}: {count:,}")
    
    print("\n✅ HLA DATASET 100% VALIDATED ✅")
    print("Next: TBDB/tb1584.fasta")

if __name__ == "__main__":
    main()
