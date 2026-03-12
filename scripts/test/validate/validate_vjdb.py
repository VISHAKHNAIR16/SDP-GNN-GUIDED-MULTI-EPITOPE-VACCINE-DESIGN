#!/usr/bin/env python3
"""
Script 5: VDJdb TCR Data Validation
File: data/VDJB/vjdb.tsv
"""

import pandas as pd
from pathlib import Path

VDJ_FILE = Path("data") / "VJDB" / "vjdb.tsv"

def validate_vjdb_tsv(filepath: Path):
    """Validate VDJdb TCR-epitope database"""
    print("\n" + "="*70)
    print("🧬 VDJB TCR DATABASE VALIDATION")
    print(f"📁 {filepath}")
    print("="*70)
    
    if not filepath.exists():
        print("❌ MISSING - Download from: http://www.vdjdb.org/")
        return None
    
    # VDJdb expected columns
    expected_cols = ['CDR3', 'V', 'J', 'Species', 'MHC A', 'MHC B', 'Epitope']
    
    try:
        df = pd.read_csv(filepath, sep='\t', low_memory=False)
        print(f"✅ LOADED: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"Columns: {list(df.columns[:10])}...")
        
        # Key column checks
        print("\n🔍 KEY COLUMNS:")
        for col in expected_cols:
            if col in df.columns:
                print(f"  ✅ {col}: {df[col].nunique():,} unique")
            else:
                print(f"  ❌ MISSING: {col}")
        
        # TCR stats
        if 'cdr3' in df.columns:
            cdr3_lengths = df['cdr3'].astype(str).str.len()
            print(f"\n🧬 CDR3 STATS:")
            print(f"  Length range: {cdr3_lengths.min()} - {cdr3_lengths.max()}")
            print(f"  Median: {cdr3_lengths.median():.0f}")
        
        # Epitope overlap with your data?
        if 'epitope' in df.columns:
            epitope_count = df['epitope'].nunique()
            print(f"  Unique epitopes: {epitope_count:,}")
        
        print("\n✅ VDJB READY FOR TCR RECOGNITION ANALYSIS")
        return df.shape
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return None

if __name__ == "__main__":
    validate_vjdb_tsv(VDJ_FILE)
