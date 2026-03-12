#!/usr/bin/env python3
"""
Fix IEDB double-header CSV files and create clean versions
Output: epitope_table_positive_clean.csv, epitope_table_negative_clean.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("data")
FILES = [
    DATA_DIR / "IEDB" / "epitope_table_positive.csv",
    DATA_DIR / "OPTIONAL" / "epitope_table_negative.csv",
]

def fix_iedb_headers(filepath: Path) -> pd.DataFrame:
    """Load IEDB double-header CSV and clean column names"""
    print(f"🔧 Processing: {filepath.name}")
    
    # Read BOTH header rows
    df = pd.read_csv(filepath, header=[0, 1], low_memory=False)
    
    # Combine multiindex columns: "Epitope|Sequence" style
    new_cols = []
    for a, b in df.columns:
        col_a = str(a).strip() if pd.notna(a) else ""
        col_b = str(b).strip() if pd.notna(b) else ""
        combined = "|".join([c for c in [col_a, col_b] if c])
        new_cols.append(combined)
    
    df.columns = new_cols
    print(f"✅ New column names (first 10): {list(df.columns[:10])}")
    
    # Find peptide column (look for "Epitope|*" or "Epitope")
    peptide_col = None
    for col in df.columns:
        if "epitope" in col.lower() and "id" not in col.lower():
            peptide_col = col
            break
    
    if peptide_col:
        print(f"🎯 Peptide column detected: '{peptide_col}'")
        # Rename to standard name
        df = df.rename(columns={peptide_col: "peptide_seq"})
        print("✅ Renamed to 'peptide_seq'")
    else:
        print("⚠️ No peptide column found")
    
    return df

def main():
    print("🔧 IEDB HEADER FIXER v1.0")
    
    for filepath in FILES:
        if not filepath.exists():
            print(f"❌ Missing: {filepath}")
            continue
        
        df = fix_iedb_headers(filepath)
        
        # Save cleaned version
        outpath = filepath.parent / f"{filepath.stem}_clean.csv"
        df.to_csv(outpath, index=False)
        print(f"💾 Saved: {outpath}")
        print(f"   Shape: {df.shape}")
        print()
    
    print("✅ All files processed! Use *_clean.csv for all future scripts.")

if __name__ == "__main__":
    main()
