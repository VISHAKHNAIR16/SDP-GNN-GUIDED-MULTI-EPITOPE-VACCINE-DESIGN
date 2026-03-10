#!/usr/bin/env python3
"""
Script 1: Basic structural validation of epitope tables
Author: Your Name | Project: GNN-Guided MDR-TB Epitope Prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Use uv's virtual environment paths
DATA_DIR = Path("data")

# Files to validate
POS_FILE = DATA_DIR / "IEDB" / "epitope_table_positive.csv"
NEG_FILE = DATA_DIR / "OPTIONAL" / "epitope_table_negative.csv"

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

def validate_epitope_file(filepath: Path, label: str) -> pd.DataFrame | None:
    """Comprehensive validation of single epitope table"""
    print(f"\n{'='*70}")
    print(f"🔍 VALIDATING {label.upper()} EPITOPES")
    print(f"📁 Path: {filepath.absolute()}")
    print('='*70)
    
    if not filepath.exists():
        print(f"❌ FILE MISSING: {filepath}")
        return None
    
    try:
        # Load with flexible options
        df = pd.read_csv(filepath, low_memory=False)
        print(f"✅ LOADED: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"📋 Columns: {list(df.columns)}")
        
        # 1. MISSING VALUES
        print("\n📊 MISSING VALUES PER COLUMN:")
        missing_summary = df.isnull().sum()
        total_missing = missing_summary.sum()
        if total_missing > 0:
            missing_pct = (missing_summary / len(df)) * 100
            issues = pd.DataFrame({
                'count': missing_summary[missing_summary > 0],
                '%': missing_pct[missing_summary > 0]
            })
            print(issues.round(2))
            print(f"⚠️  Total missing cells: {total_missing:,} ({total_missing/len(df)/df.shape[1]*100:.1f}%)")
        else:
            print("✅ No missing values")
        
        # 2. DUPLICATES
        print("\n🔍 DUPLICATES:")
        row_dups = df.duplicated().sum()
        print(f"Duplicate rows: {row_dups}")
        
        # Auto-detect peptide column
        peptide_col = None
        peptide_candidates = ['peptide', 'sequence', 'epitope', 'Epitope.Sequence', 'Peptide']
        for col in peptide_candidates:
            if col in df.columns:
                peptide_col = col
                break
        
        if peptide_col:
            pep_series = df[peptide_col].dropna()
            pep_dups = pep_series.duplicated().sum()
            print(f"Duplicate {peptide_col}: {pep_dups:,}")
            print(f"Unique {peptide_col}: {pep_series.nunique():,}")
            
            # 3. SEQUENCE VALIDATION
            print("\n🧬 SEQUENCE VALIDATION:")
            invalid_chars = pep_series.apply(lambda x: any(c.upper() not in VALID_AA for c in str(x)))
            invalid_count = invalid_chars.sum()
            print(f"Invalid characters: {invalid_count:,}")
            
            lengths = pep_series.str.len()
            print(f"Length stats: min={lengths.min()}, max={lengths.max()}, median={lengths.median():.1f}")
            
            if invalid_count > 0:
                print("Examples with invalid chars:")
                print(pep_series[invalid_chars].head(3).tolist())
        else:
            print("⚠️ No peptide column detected (looked for:", peptide_candidates, ")")
        
        print(f"\n✅ {label} file validation COMPLETE")
        return df
        
    except Exception as e:
        print(f"❌ ERROR loading {filepath}: {e}")
        return None

def main():
    print("🚀 TB EPITOPE VALIDATION PIPELINE v1.0")
    print(f"Working dir: {Path.cwd()}")
    print("-" * 70)
    
    pos_df = validate_epitope_file(POS_FILE, "POSITIVE")
    neg_df = validate_epitope_file(NEG_FILE, "NEGATIVE")
    
    if pos_df is not None and neg_df is not None:
        print("\n" + "="*70)
        print("📊 GRAND SUMMARY")
        print("="*70)
        print(f"Positive: {pos_df.shape[0]:,} | Negative: {neg_df.shape[0]:,}")
        print(f"Total: {pos_df.shape[0] + neg_df.shape[0]:,}")
        print("🎉 ALL FILES VALIDATED SUCCESSFULLY")
    
    print("\n💾 Results saved to console. Next: detailed sequence analysis.")

if __name__ == "__main__":
    main()
