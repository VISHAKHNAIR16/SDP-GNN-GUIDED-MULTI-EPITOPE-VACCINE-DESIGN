#!/usr/bin/env python3
"""
Script 2: Detailed sequence validation for CLEAN IEDB files
Works with peptide_seq column from fix_iedb_headers.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path("data")
POS_CLEAN = DATA_DIR / "IEDB" / "epitope_table_positive_clean.csv"
NEG_CLEAN = DATA_DIR / "OPTIONAL" / "epitope_table_negative_clean.csv"

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

def validate_sequences(filepath: Path, label: str):
    """Full sequence validation"""
    print(f"\n{'='*80}")
    print(f"🧬 DETAILED SEQUENCE VALIDATION: {label}")
    print(f"📁 {filepath}")
    print('='*80)
    
    df = pd.read_csv(filepath)
    
    # Peptide column is now guaranteed
    assert 'peptide_seq' in df.columns, "Run fix_iedb_headers.py first!"
    
    peptides = df['peptide_seq'].dropna().astype(str)
    print(f"Total peptides: {len(peptides):,}")
    
    # 1. LENGTH ANALYSIS
    lengths = peptides.str.len()
    print(f"\n📏 LENGTH STATS:")
    print(f"  Range: {lengths.min()} - {lengths.max()}")
    print(f"  Median: {lengths.median():.1f}")
    print(f"  25th/75th: {lengths.quantile(0.25):.1f} / {lengths.quantile(0.75):.1f}")
    
    # 2. SEQUENCE QUALITY
    print(f"\n🔬 QUALITY CHECKS:")
    upper_peptides = peptides.str.upper()
    
    # Invalid characters
    invalid_mask = upper_peptides.apply(lambda x: any(c not in VALID_AA for c in x))
    invalid_count = invalid_mask.sum()
    print(f"  Invalid AA chars: {invalid_count:,} ({invalid_count/len(peptides)*100:.2f}%)")
    
    # Non-standard (B, J, O, U, X, Z)
    nonstd_mask = upper_peptides.apply(lambda x: any(c in 'BJOUXZ' for c in x))
    nonstd_count = nonstd_mask.sum()
    print(f"  Non-standard AA: {nonstd_count:,} ({nonstd_count/len(peptides)*100:.2f}%)")
    
    # Empty/NaN after dropna
    empty_mask = (lengths == 0)
    print(f"  Empty sequences: {empty_mask.sum():,}")
    
    # Length outliers (epitopes typically 8-20)
    short_outliers = (lengths < 8).sum()
    long_outliers = (lengths > 20).sum()
    print(f"  Length outliers: <8aa={short_outliers}, >20aa={long_outliers}")
    
    # 3. COMMON LENGTHS (for presentation)
    length_counts = lengths.value_counts().sort_index()
    print(f"\n📊 MOST COMMON LENGTHS:")
    print(length_counts.head(10))
    
    # Save clean stats
    clean_stats = {
        'total': len(peptides),
        'valid_length': ((lengths >= 8) & (lengths <= 20)).sum(),
        'valid_aa': (~invalid_mask).sum(),
        'common_lengths': length_counts.to_dict()
    }
    
    print(f"\n✅ CLEAN SUMMARY: {clean_stats['valid_length']:,} high-quality peptides ready for modeling")
    return clean_stats, lengths

def main():
    print("🧬 TB EPITOPE SEQUENCE VALIDATOR v2.0")
    
    pos_stats, pos_lengths = validate_sequences(POS_CLEAN, "POSITIVE")
    neg_stats, neg_lengths = validate_sequences(NEG_CLEAN, "NEGATIVE")
    
    print("\n" + "="*80)
    print("🎯 GRAND SEQUENCE SUMMARY")
    print("="*80)
    print(f"Positive clean: {pos_stats['valid_length']:,}/{pos_stats['total']:,}")
    print(f"Negative clean: {neg_stats['valid_length']:,}/{neg_stats['total']:,}")
    print(f"Total clean: {pos_stats['valid_length'] + neg_stats['valid_length']:,}")
    print("\n🎉 SEQUENCE VALIDATION COMPLETE - DATA READY FOR EDA!")
    
    # Quick length plot
    fig, ax = plt.subplots(figsize=(10, 4))
    pd.Series(pos_lengths).hist(bins=30, alpha=0.7, label='Positive', ax=ax)
    pd.Series(neg_lengths).hist(bins=30, alpha=0.7, label='Negative', ax=ax)
    ax.set_xlabel('Peptide Length')
    ax.set_ylabel('Count')
    ax.set_title('Peptide Length Distribution')
    ax.legend()
    plt.tight_layout()
    plt.savefig('peptide_length_distribution.png', dpi=300, bbox_inches='tight')
    print("\n📈 Length distribution saved: peptide_length_distribution.png")
    plt.show()

if __name__ == "__main__":
    main()
