#!/usr/bin/env python3
"""
Debug: Print first 5 headers from each HLA FASTA
"""

from pathlib import Path
from Bio import SeqIO
import os

HLA_FILES = "data/HLA/hla_gen.fasta"

def inspect_fasta(filepath):
    """Basic inspection of a FASTA file."""
    if not filepath or not os.path.exists(filepath):
        print("FASTA file not found.")
        return

    sequence_count = 0
    line_count = 0
    current_seq_length = 0
    sequence_lengths = []
    header_chars = 0

    print(f"\n--- Inspecting FASTA file: {filepath} ---")
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line_count += 1
                line = line.strip()
                if not line: # Skip empty lines
                    continue

                if line.startswith('>'):
                    # Finished previous sequence, record its length
                    if current_seq_length > 0:
                        sequence_lengths.append(current_seq_length)
                        current_seq_length = 0
                    sequence_count += 1
                    # Optional: print first few headers to see format
                    if sequence_count <= 5:
                        print(f"Header {sequence_count}: {line[:100]}...") # Print first 100 chars
                    header_chars += len(line)
                else:
                    # This is a sequence line
                    current_seq_length += len(line)
                    # Basic check: ensure it contains only valid IUPAC nucleotide codes (optional)
                    # valid_chars = set("ACGTURYKMSWBDHVN-")
                    # if not all(c in valid_chars for c in line.upper()):
                    #     print(f"Warning: Non-standard character at line {line_count}")

            # Add the last sequence length
            if current_seq_length > 0:
                sequence_lengths.append(current_seq_length)

        # --- Summary Report ---
        print(f"\n--- Inspection Complete ---")
        print(f"Total lines in file: {line_count}")
        print(f"Total sequences (headers starting with '>'): {sequence_count}")
        print(f"Approximate total header characters: {header_chars}")

        if sequence_lengths:
            print(f"Sequence length stats (in bases):")
            print(f"  - Min length: {min(sequence_lengths)}")
            print(f"  - Max length: {max(sequence_lengths)}")
            print(f"  - Average length: {sum(sequence_lengths)/len(sequence_lengths):.2f}")
            # Check if lengths vary significantly (normal for genomic sequences)
            if len(set(sequence_lengths)) > 10:
                 print("  - Note: Sequence lengths vary, as expected for genomic DNA.")
        else:
            print("No sequence data found after headers.")

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        print("This could indicate file corruption.")

# --- Run the inspection ---
# Use the fasta_file_path variable from the extraction step
if 'fasta_file_path' in locals() and fasta_file_path:
    inspect_fasta(fasta_file_path)
else:
    print("Please ensure the file is decompressed and provide the correct path.")
    # You can also manually set the path:
    # inspect_fasta('/path/to/your/extracted/hla_gen.fasta')



inspect_fasta(HLA_FILES)