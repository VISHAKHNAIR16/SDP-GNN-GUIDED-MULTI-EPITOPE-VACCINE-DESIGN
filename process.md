# Full Process one by one 


Validation scripts


| Script name (logical)             | Purpose (what it does)                                                                                                                        | Key outputs (for presentation)                                                                                                    |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| validate_epitope_tables.py        | Check epitope_table_positive.csv and epitope_table_negative.csv for missing values, invalid characters, inconsistent columns, and duplicates. | Summary table of missing values, counts of invalid sequences, duplicate counts; simple bar chart of unique vs duplicate epitopes. |
| validate_sequence_content.py      | Validate peptide sequences (positive and negative) contain only valid amino acid letters and reasonable length ranges.                        | Histogram of peptide length distribution; count of invalid characters; list of extreme outliers.                                  |
| validate_hla_alleles.py           | Check HLA allele columns for proper nomenclature (e.g., HLA-A*02:01), missing values, and per-allele sample counts.                           | Table of allele frequencies; bar chart of top N alleles by sample count.                                                          |
| validate_label_consistency.py     | Detect sequences that appear in both positive and negative tables or with conflicting labels.                                                 | Table with conflicting sequences and counts; total number of conflicts.                                                           |
| validate_vdjb_links.py (optional) | If TCR/VDJ data (vjdb.tsv) links to epitopes, ensure IDs and sequences match across files.                                                    | Count of matched vs unmatched IDs; percentage of linkage completeness.                                                            |


EDA scripts


| Script name (logical)                                 | Purpose (what it does)                                                                         | Key graphs / outputs                                                                           |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| eda_class_balance.py                                  | Explore overall class balance between positive and negative epitopes.                          | Bar chart of positive vs negative counts; ratio printed.                                       |
| eda_peptide_length.py                                 | Analyze length distribution of epitopes for each class.                                        | Overlaid histograms / density plots of peptide lengths per class; boxplots of length by class. |
| eda_amino_acid_composition.py                         | Compute amino acid frequencies for positive and negative sets.                                 | Side-by-side bar charts of amino acid frequencies in positives vs negatives.                   |
| eda_position_specific_patterns.py                     | For peptides of a dominant length (e.g., 9-mers), examine per-position amino acid preferences. | Heatmap of amino acid frequency by position (sequence logo–like view).                         |
| eda_hla_distribution.py                               | Study distribution of HLA alleles and how many peptides each allele has.                       | Bar chart of top HLA alleles; cumulative coverage plot (alleles vs fraction of data).          |
| eda_hla_epitope_matrix.py                             | Create HLA–epitope count matrix to see which alleles share epitopes and overall connectivity.  | Bipartite degree distributions; heatmap of counts (HLA on one axis, maybe grouped).            |
| eda_vdjb_tcr_stats.py (optional)                      | Describe TCR/VDJ data: clonotype counts, CDR3 length, gene usage.                              | Histogram of CDR3 length; bar chart of top V/J genes.                                          |
| eda_feature_corr_preview.py (later, after embeddings) | Once you compute sequence/PLM features, check correlations and basic structure.                | Correlation heatmap of engineered features; basic PCA scatterplot.                             |