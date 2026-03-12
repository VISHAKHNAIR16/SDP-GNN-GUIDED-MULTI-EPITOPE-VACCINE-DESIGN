#!/usr/bin/env python3
"""
Script 4: TBDB MDR-TB Antigen Validation
File: data/TBDB/tb1584.fasta (Rv0248c or similar)
"""

from pathlib import Path
from Bio import SeqIO

TB_FILE = Path("data") / "TBDB" / "tb1584.fasta"
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

def validate_tb_antigen(filepath: Path):
    """Validate TB antigen FASTA"""
    print("\n" + "="*70)
    print("🦠 MDR-TB ANTIGEN VALIDATION (TBDB)")
    print(f"📁 {filepath}")
    print("="*70)
    
    if not filepath.exists():
        print("❌ MISSING - Download TB antigen (Rv0248c recommended)")
        print("UniProt: P9WHE3 or similar MDR-TB protein")
        return None
    
    try:
        records = list(SeqIO.parse(filepath, "fasta"))
        print(f"✅ LOADED: {len(records)} protein(s)")
        
        for i, rec in enumerate(records):
            seq = str(rec.seq).upper()
            print(f"\nProtein {i+1}: '{rec.id}'")
            print(f"  Length: {len(seq)} AA")
            print(f"  First 50 AA: {seq[:50]}")
            
            invalid_chars = set(seq) - VALID_AA
            print(f"  Invalid AA: {invalid_chars}")
            
            if len(seq) < 50:
                print("  ⚠️  Very short - check source")
        
        print("\n✅ TB ANTIGEN READY FOR EPITOPE SCANNING")
        return len(records)
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return None

if __name__ == "__main__":
    validate_tb_antigen(TB_FILE)
