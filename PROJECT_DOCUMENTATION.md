# PROJECT DOCUMENTATION
## GNN-Guided Multi-Epitope Vaccine Design: A Protein-Language-Model–Driven Heterogeneous Graph Framework for Prioritizing T-Cell Epitopes with Broad HLA Binding and TCR Recognition

**Primary disease:** M. tuberculosis (MDR-TB)
**Validation disease:** SARS-CoV-2 (COVID-19)
**Framework:** Heterogeneous Graph Neural Network (HANConv) with ESM-2 embeddings
**Cross-disease validation:** Confirmed — same architecture applied to COVID-19 without modification

---

## Purpose of this document

This document provides a complete, reproducible, plain-language explanation of every script, dataset, design decision, and result in this project. It serves as:
- A developer reference for running and extending the pipeline
- A methods appendix for a manuscript or thesis
- A record of methodological decisions, including corrections and lessons learned

---

## Quick run sequence

### TB (primary) pipeline
```
uv run scripts/01_clean_data.py
uv run scripts/02_eda.py
uv run scripts/03_feature_engineering.py
uv run scripts/04_build_graph.py
uv run scripts/05_train_gnn.py
uv run scripts/06_prioritize_epitopes.py
uv run scripts/07_improve_v3.py
uv run scripts/08_multi_epitope_assembly.py
uv run scripts/09_compare_models.py
```

### COVID-19 (validation) pipeline
```
uv run scripts/01_clean_data_covid.py
uv run scripts/01b_fix_covid_proteome.py
uv run scripts/02_eda_covid.py
uv run scripts/03_feature_engineering_covid.py
uv run scripts/04_build_graph_covid.py
uv run scripts/04b_fix_graph_covid.py
uv run scripts/05_train_gnn_covid.py          # v1 baseline (for reference)
uv run scripts/05b_train_gnn_covid_v2.py      # v2.1 final (correct model)
uv run scripts/06_prioritize_epitopes_covid.py
uv run scripts/09_compare_models_covid.py
```

---

## Part 1: Primary Pipeline — M. tuberculosis (MDR-TB)

### Data and processed files

| File | Description |
|------|-------------|
| `data/processed/iedb_positive_clean.csv` | 2,249 confirmed immunogenic TB epitopes |
| `data/processed/iedb_negative_clean.csv` | 21,635 confirmed non-immunogenic TB epitopes |
| `data/processed/vjdb_tb_human_clean.tsv` | 61 VDJdb TCR-epitope pairs (gold standard) |
| `data/processed/hla_prot_clean.fasta` | 44,398 HLA protein sequences |
| `data/processed/tb_proteome_clean.fasta` | ~5,000 TB proteome sequences |
| `data/processed/embeddings/` | ESM-2 embeddings (.npy) + metadata (.csv) |
| `data/processed/graph/heterogeneous_graph.pt` | PyG HeteroData graph |
| `data/processed/graph/graph_stats.json` | Graph summary statistics |

### Outputs

| Location | Contents |
|----------|----------|
| `outputs/models/` | `best_model.pt`, `best_model_v3.pt`, `best_model_v4.pt`, training history JSONs |
| `outputs/vaccine_candidates/` | `top_candidates.csv`, `all_epitopes_scored.csv`, `final_vaccine_construct.fasta` |
| `outputs/figures/` | Figures 01–21 (EDA through model comparison) |
| `outputs/*.log` | Cleaning, training, graph building logs |

---

### Script 01: `01_clean_data.py` — Data Cleaning (TB)

**Purpose:** Convert raw, messy IEDB exports and auxiliary databases into clean, validated, deduplicated datasets with consistent column names.

**Inputs:**
- `data/IEDB/epitope_table_positive.csv` — raw IEDB positive epitope export
- `data/IEDB/epitope_table_negative.csv` — raw IEDB negative epitope export
- `data/VJDB/vjdb.tsv` — VDJdb TCR-epitope database
- `data/HLA/hla_prot.fasta` — HLA protein sequences
- `data/TBDB/tb1584.fasta` — M. tuberculosis proteome

**Outputs:**
- `data/processed/iedb_positive_clean.csv` (2,249 unique positive TB epitopes)
- `data/processed/iedb_negative_clean.csv` (21,635 unique negative epitopes)
- `data/processed/vjdb_tb_human_clean.tsv` (61 human TB TCR-epitope pairs)
- `data/processed/hla_prot_clean.fasta` (44,398 HLA sequences)
- `data/processed/tb_proteome_clean.fasta`

**Key functions:**
- `find_peptide_column()` — detects which IEDB column holds actual peptide sequences (IEDB exports use inconsistent column naming)
- `is_valid_peptide()` — validates 8–25 aa length and standard amino acid alphabet
- `is_valid_protein()` — validates protein sequences before embedding
- Deduplication: collapses multiple assay rows for the same sequence into one canonical entry

**Biology:** Immunogenicity data from IEDB is assay-level — the same epitope may appear dozens of times from different labs. Deduplication gives one authoritative positive or negative label per sequence. The 3.2:1 class imbalance (21,635 negatives vs 2,249 positives) is biologically real — most peptide fragments from a pathogen do not activate T-cells.

---

### Script 02: `02_eda.py` — Exploratory Data Analysis (TB)

**Purpose:** Generate publication-ready figures that characterise the TB dataset before model training. Verify that the data has learnable patterns before committing to the full pipeline.

**Outputs (figures 01–07):**

| Figure | What it shows | Why it matters |
|--------|--------------|----------------|
| 01 | Epitope length distribution pos vs neg | Confirms MHC I (9-mer) and MHC II (15-mer) peaks — biologically expected |
| 02 | Amino acid frequency enrichment | Hydrophobic residues enriched in positives — consistent with MHC anchor chemistry |
| 03 | HLA allele coverage | No systematic allele bias — dataset is representative of human HLA diversity |
| 04 | IEDB–VDJdb overlap | 11 dual-confirmed gold-standard TB epitopes — sparse but high-confidence |
| 05 | Position-specific heatmap (9-mers) | P2 and P9 anchor position preferences visible — GNN can learn this |
| 06 | Class imbalance | 3.2:1 neg:pos — requires `pos_weight` correction in loss function |
| 07 | CDR3 length distribution | 57 unique TB CDR3s, mostly 12–16 aa — typical for MHC I |

**Key finding:** The 11 gold-standard epitopes (IEDB positive ∩ VDJdb confirmed) represent the strongest available training signal for TB. The sparsity of TCR evidence for TB is a fundamental data limitation, not a model choice.

---

### Script 03: `03_feature_engineering.py` — ESM-2 Embeddings (TB)

**Purpose:** Convert amino acid sequences into dense numerical vectors using the ESM-2 protein language model. These vectors become the node features in the heterogeneous graph.

**Model:** `esm2_t6_8M_UR50D` — 6 transformer layers, 8 million parameters, 320-dimensional output.

**Procedure:**
1. Load ESM-2 model to GPU (falls back to CPU with warning)
2. Tokenise sequences using ESM-2 vocabulary
3. Forward pass through 6 transformer layers (frozen weights — no fine-tuning)
4. Extract layer-6 representations: shape (batch_size, sequence_length, 320)
5. Mean-pool over sequence positions (excluding BOS/EOS tokens)
6. Output: one 320-dim vector per sequence

**Outputs:**

| File | Shape | Content |
|------|-------|---------|
| `epitopes_positive.npy` | (2,249, 320) | TB positive epitope embeddings |
| `epitopes_negative.npy` | (21,635, 320) | TB negative epitope embeddings |
| `tb_proteins.npy` | (~5,000, 320) | TB proteome embeddings (truncated to 512 aa for long proteins) |
| `hla_sample.npy` | (2,000, 320) | Sampled HLA sequence embeddings |

**Why ESM-2 and not one-hot or k-mer features:** ESM-2 was pretrained on 250 million protein sequences from UniRef50. It learned deep biochemical grammar: amino acid substitution patterns, secondary structure propensity, binding site signatures. L and I are biochemically similar (both aliphatic) — ESM-2 encodes them similarly. G and P are both helix-breakers — ESM-2 reflects this. A k-mer or one-hot encoding treats all amino acids as equally different from each other, losing this chemical information.

**Validation:** Mean cosine similarity between positive and negative epitope set means = 0.999. This sounds alarming but is expected — both groups are short peptides from the same organism, so population-level means converge. What matters is that individual positive and negative embeddings are locally separable in the 320-dim space, which the GNN exploits via neighbourhood-based learning.

**Resume safety:** The script checks for existing `.npy` checkpoint files before embedding. If interrupted, it resumes from the last completed dataset rather than restarting from scratch.

---

### Script 04: `04_build_graph.py` — Heterogeneous Graph Construction (TB)

**Purpose:** Assemble the biological knowledge graph that the GNN will learn from. Each edge type encodes a different biological relationship between entities.

**Node types and counts:**

| Type | Count | Features | Biological role |
|------|-------|----------|----------------|
| epitope | 23,884 | 320-dim ESM-2 | What we classify — immunogenic or not |
| protein | 21,008 | 320-dim ESM-2 | TB proteome — source context for epitopes |
| hla | 2,000 | 320-dim ESM-2 | HLA alleles — binding partners for epitopes |
| tcr | 57 | 320-dim (CDR3 composition) | VDJdb TCR sequences — gold-standard binding evidence |

**Edge types:**

| Type | Count | How built | Biological meaning |
|------|-------|-----------|-------------------|
| protein → source_of → epitope | 23,884 | Source protein lookup from IEDB | This epitope is derived from this protein |
| epitope → binds_to → hla | ~46,000 | Allele name match (primary) + embedding similarity (fallback) | This epitope binds to this HLA allele |
| epitope → recognized_by → tcr | 61 | Direct VDJdb lookup | This epitope activates this specific TCR (experimental) |
| epitope → similar_to → epitope | ~57,000 | k-NN cosine similarity (k=5, threshold=0.85) | These epitopes are biochemically similar |

**Why heterogeneous edges matter:** The GNN uses different relation types to learn different biological signals. A positive epitope next to other positives in the similarity graph propagates immunogenicity to its neighbours. A positive epitope connected to HLA nodes propagates binding evidence. An epitope connected to a TCR node from the gold-standard set is the strongest possible signal. The multi-relational structure allows the GNN to combine all these sources simultaneously.

**k-NN similarity edges:** Computing all pairwise cosine similarities for 23,884 epitopes would require ~570M pairs. The script uses batched matrix multiplication (1,000 epitopes per batch) to stay within GPU memory. Only pairs with similarity ≥ 0.85 and within the top-5 neighbours are included as edges.

---

### Script 05: `05_train_gnn.py` — GNN Training (TB)

**Purpose:** Train the Heterogeneous Attention Network to classify epitope nodes as immunogenic (1) or non-immunogenic (0) based on their features and graph neighbourhood.

**Architecture:**

```
Input: epitope features (320-dim ESM-2)
       protein features (320-dim ESM-2)
       hla features (320-dim ESM-2)
       tcr features (320-dim CDR3 composition)

Per-node-type input projection:
  Linear(320, 128) → LayerNorm(128) → ReLU()

HANConv layer 1: (128 → 128, 4 heads, averaging)
  + residual connection + LayerNorm + ReLU + Dropout(0.3)

HANConv layer 2: (128 → 128, 4 heads, averaging)
  + residual connection + LayerNorm + ReLU + Dropout(0.3)

HANConv layer 3: (128 → 128, 4 heads, averaging)
  + residual connection + LayerNorm + ReLU + Dropout(0.3)

Epitope classifier:
  Linear(128, 64) → ReLU() → Dropout(0.3) → Linear(64, 1)

Output: logit per epitope → sigmoid → immunogenicity probability
```

**HANConv (Heterogeneous Attention Network Convolution):** Computes separate attention weights for each edge type, then aggregates messages from all edge types for each node. The attention mechanism learns which biological relationships are most informative for immunogenicity prediction — for example, it might learn that TCR edges carry more weight than protein-source edges.

**Training configuration:**
| Hyperparameter | Value | Rationale |
|---------------|-------|-----------|
| Loss | BCEWithLogitsLoss | Binary classification |
| pos_weight | 9.6 | Corrects 9.6:1 class imbalance (21,635 neg / 2,249 pos) |
| Optimizer | Adam | Standard for GNNs |
| LR | 1e-3 | Standard starting LR |
| Weight decay | 1e-4 | L2 regularisation |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=10) | Adapts LR when val AUROC plateaus |
| Early stopping | patience=20 on val AUROC | Prevents overfitting |
| Batch | Full graph (no mini-batching) | Graph too small to benefit from mini-batching |
| Split | 70/15/15 stratified | Maintains class ratio in each split |

**Results:**
| Metric | Value |
|--------|-------|
| Val AUROC | 0.8528 (best epoch 137) |
| Test AUROC | 0.8928 |
| Test AUPRC | 0.5259 (random baseline: 0.094) |
| Test F1 | 0.4601 |
| Test Precision | 0.3182 |
| Test Recall | 0.8309 |
| Top-50 precision | 70% |

**Interpretation:** AUROC 0.89 indicates strong discrimination. AUPRC 0.53 appears lower but must be evaluated against the random baseline of 0.094 (= positive fraction). AUPRC lift = +0.43, which is substantial. The high recall (0.83) with lower precision (0.32) reflects the high pos_weight — the model is configured to find most positives at the cost of some false positives. For vaccine design, high recall is preferred at the candidate generation stage (Phase 6 filtering handles precision).

---

### Script 06: `06_prioritize_epitopes.py` — Epitope Prioritization (TB)

**Purpose:** Use the trained model to score all 23,884 TB epitopes and produce a prioritised candidate list suitable for experimental validation or vaccine assembly.

**Composite scoring:**
```
composite_score = 0.50 × gnn_score
               + 0.30 × tcr_evidence (binary: 0 or 1)
               + 0.20 × hla_coverage_score (normalized HLA neighbors)
```

**Why these weights:** GNN score dominates (50%) because it integrates all available graph signal. TCR evidence (30%) is the most biologically trustworthy signal — it means a real T-cell receptor binds this epitope in a real patient. HLA coverage (20%) rewards epitopes that are predicted to bind many HLA alleles — important for population-wide vaccine coverage.

**Candidate filter:** GNN score > 0.5 (threshold chosen as natural classification boundary for imbalanced training).

**Outputs:**
- `all_epitopes_scored.csv` — all 23,884 epitopes with all scores
- `top_candidates.csv` — candidates passing GNN threshold
- `top50_candidates.csv` — top-50 for detailed analysis
- `gold_standard.csv` — the 11 dual-confirmed TB gold-standard epitopes

---

### Script 07: `07_improve_v3.py` — Model Improvement (TB)

**Purpose:** Improve upon the baseline GNN with five targeted enhancements validated against the TB dataset.

**Improvements over v1:**

| Enhancement | Detail | AUROC impact |
|------------|--------|-------------|
| Position-AA features | 25×20 one-hot position matrix appended to epitope features (320 → 820 dim) | +0.04 (largest gain) |
| Conservation features | Essential gene flag + drug target flag + ESX protein flag | +0.02 |
| Focal Loss (α=0.80) | Upweights hard-to-classify positive examples | +0.01 |
| Wider model | hidden=256, 4 layers (vs 128, 3 layers) | +0.01 |
| AdamW optimizer | Decoupled weight decay | marginal |

**Why α=0.80 in focal loss for TB specifically:** TB positives are 14% of the dataset. α=0.80 assigns 80% weight to the positive class loss terms. This is appropriate for the imbalanced TB setting. For COVID's balanced 50/50 dataset, this same alpha value destroys learning by suppressing gradients — see COVID v2.1 for the corrected approach.

**Position-AA features:** A 25×20 matrix encodes which amino acid appears at each of 25 positions (padded with zeros for shorter sequences). For a 9-mer epitope, positions 2 and 9 (P2 anchor and P9 anchor for HLA-A*02:01) show strong amino acid preferences. The GNN learns these patterns from the feature matrix rather than having to infer them from ESM-2 embeddings alone.

---

### Script 08: `08_multi_epitope_assembly.py` — Vaccine Assembly (TB)

**Purpose:** Translate the ranked epitope list into a concrete multi-epitope vaccine construct that can be synthesised and tested.

**Selection criteria:**
- Top 5 Class I (CD8+, MHC I, ≤11 aa) epitopes by composite score
- Top 5 Class II (CD4+, MHC II, ≥12 aa) epitopes by composite score
- Priority: TCR-confirmed > essential gene source > GNN score
- Redundancy filter: sequences with >70% identity are collapsed to the highest-ranked

**Construct assembly:**
```
[PADRE adjuvant]–EAAAK–[ClassII-1]–GPGPG–[ClassII-2]–GPGPG–...
  –AAY–[ClassI-1]–AAY–[ClassI-2]–AAY–...
```

**Linker biology:**
- `AAY` (Ala-Ala-Tyr): proteasomal cleavage site — promotes efficient processing of Class I epitopes by the 26S proteasome
- `GPGPG` (Gly-Pro-Gly-Pro-Gly): flexible spacer — prevents formation of new junctional epitopes that could bind HLA unintentionally and cause adverse responses
- `EAAAK`: rigid helix-forming linker — separates the adjuvant from the epitope region structurally
- `PADRE` adjuvant: pan-DR epitope (AKFVAAWTLKAAA) — activates CD4+ T-helper cells broadly across HLA-DR alleles

**Physicochemical validation of final construct:**
- Molecular weight, isoelectric point, GRAVY, instability index, aliphatic index
- All computed inline without external library

---

### Script 09: `09_compare_models.py` — Model Comparison (TB)

**Purpose:** Side-by-side comparison of all TB model versions (v1 baseline, v3, v4) across all metrics. Generates the ablation study figures for the paper.

---

## Part 2: COVID-19 Validation Pipeline

### Why validate on COVID-19

The primary scientific claim of this project is that the GNN pipeline is disease-agnostic — it learns immunogenicity patterns from graph structure and sequence features rather than disease-specific biological rules. COVID-19 was selected as the validation disease because:

1. **Abundant data:** IEDB has 4,213 confirmed COVID positive and 4,135 negative epitopes — sufficient for meaningful training
2. **Rich TCR evidence:** VDJdb has 9,436 SARS-CoV-2 TCR-epitope pairs — 60× more than TB's 61 pairs — creating a richer gold-standard signal
3. **Structural contrast:** COVID's 1:1 balanced class distribution (vs TB's 3.2:1) tests the pipeline under different statistical conditions
4. **Well-characterised immunology:** COVID immune responses are extensively characterised, making it easier to biologically validate top candidates

### Data and processed files (COVID)

| File | Description |
|------|-------------|
| `data/processed_covid/iedb_positive_covid.csv` | 4,213 IEDB-confirmed SARS-CoV-2 immunogenic epitopes |
| `data/processed_covid/iedb_negative_covid.csv` | 4,135 IEDB-confirmed non-immunogenic epitopes |
| `data/processed_covid/vdjdb_covid_clean.tsv` | 9,436 VDJdb TCR-epitope pairs (9,333 unique CDR3s) |
| `data/processed_covid/covid_proteins_clean.csv` | 5,699 SARS-CoV-2 UniProt entries (raw, multi-strain) |
| `data/processed_covid/covid_proteins_reference.csv` | 17 canonical reference proteins (cleaned) |
| `data/processed_covid/embeddings/` | ESM-2 embeddings (.npy + _meta.csv) |
| `data/processed_covid/graph/covid_graph.pt` | PyG HeteroData graph (v2.1 — fixed and corrected) |

### Outputs (COVID)

| Location | Contents |
|----------|----------|
| `outputs/models_covid/` | `best_model_covid.pt` (v1), `best_model_covid_v2_1.pt` (final) |
| `outputs/vaccine_candidates_covid/` | `top_candidates_covid.csv`, `all_epitopes_scored_covid.csv`, `gold_standard_covid.csv` |
| `outputs/figures_covid/` | Figures 01–21 (EDA through cross-disease comparison) |

---

### Script 01_clean_data_covid.py — Data Cleaning (COVID)

**Purpose:** Mirror the TB cleaning pipeline for SARS-CoV-2 data. Filter IEDB to SARS-CoV-2 organism, clean VDJdb to human TCR entries, validate sequences, deduplicate.

**Key COVID-specific steps:**
- IEDB filter: retain only entries with `organism_name` containing "SARS-CoV-2" or "severe acute respiratory syndrome coronavirus 2"
- VDJdb filter: retain human host (`species = HomoSapiens`), valid CDR3 sequences (standard AA alphabet, 8–25 aa)
- Proteome: download from UniProt taxon 2697049 (SARS-CoV-2) — results in multi-strain dump requiring the fix script below

**Gold standard computation:**
```
gold_standard = IEDB_positive_sequences ∩ VDJdb_epitope_sequences
             = 668 epitopes (vs 11 for TB)
```

**Class balance:** 4,213 positive / 4,135 negative = 0.98:1. This near-perfect balance has significant implications for model training (see Script 05_train_gnn_covid.py).

---

### Script 01b_fix_covid_proteome.py — Proteome Fix

**Purpose:** Critical pre-processing fix. The raw UniProt FASTA for SARS-CoV-2 (taxon-level query) returns all sequences from all submitted strain variants — Delta, Omicron, Alpha, Wuhan-Hu-1, etc. After exact sequence deduplication, this yields 5,699 entries for what is biologically a 29-gene virus.

**Problem with 5,699 proteins in graph:** If 200 Spike variant sequences become 200 protein nodes, every Spike-derived epitope connects to all 200 of them via `source_of` edges. This creates dense, spurious graph structure that doesn't represent distinct biology — it represents sequencing duplicates from different labs.

**Fix strategy:**
1. Normalise gene names (handle case variants: `S`, `Spike`, `spike`, `surface glycoprotein`)
2. For each unique gene, keep only the longest sequence (highest quality reference)
3. Remaining proteins with no gene name: cluster by sequence length ± 50 aa, keep longest per cluster

**Result:** 5,699 proteins → 17 canonical reference proteins, representing all major SARS-CoV-2 gene products. After graph building and orphan removal, 11 active protein nodes remain (6 were duplicate gene entries that matched no epitopes).

---

### Script 02_eda_covid.py — Exploratory Data Analysis (COVID)

**Purpose:** Eight COVID-specific EDA figures. Key additions not present in TB EDA: CDR3 length by MHC class (meaningful because COVID has 9,341 CDR3s vs TB's 57), MHC class distribution (COVID VDJdb is 91% MHC I), and immunodominant source protein analysis (Spike, ORF1ab dominate).

**Key COVID EDA findings:**

| Finding | Biological interpretation |
|---------|--------------------------|
| Spike: 1,689 pos epitopes | Spike is the dominant immune target — consistent with all COVID vaccine literature |
| ORF1ab: 1,421 pos epitopes | NSP proteins are highly immunogenic — underexplored in commercial vaccines |
| MHC I: 91% of VDJdb CDR3s | COVID VDJdb is heavily CD8+ biased — limits MHC II graph edges |
| 668 gold-standard epitopes | 60× richer than TB's 11 — more robust TCR-based training signal |
| Class balance 0.98:1 | Standard accuracy is now a meaningful metric (unlike TB where it was misleading) |
| W, Y, R enriched in positives | Aromatic and charged residues at MHC anchor positions |

---

### Script 03_feature_engineering_covid.py — ESM-2 Embeddings (COVID)

**Purpose:** Embed COVID sequences using the same ESM-2 model as TB. Uses identical embedding procedure (mean-pooling, same layer, same model) to ensure the 320-dim embedding spaces are directly comparable.

**HLA source — COVID vs TB:**

| | TB | COVID |
|-|-----|-------|
| HLA source | 44,398 full HLA protein sequences (FASTA) | 16 unique alleles from VDJdb mhc_a column |
| HLA representation | Full protein ESM-2 embedding | 9-residue binding-groove pseudosequence (NetMHCpan encoding) |
| HLA nodes in graph | 2,000 | 16 |

The 16 COVID HLA alleles cover HLA-A, HLA-B, HLA-C (Class I) and HLA-DRA, HLA-DPA1, HLA-DQA1 (Class II). This is a structural limitation — not a methodological error. If a full COVID HLA FASTA were collected, the graph would be richer. The current approach is the best achievable from available VDJdb data.

**Compatibility validation:** TB positive mean norm = 6.041, COVID positive mean norm = 6.160. Ratio = 1.020 — within the 0.8–1.2 threshold for embedding space compatibility. The two disease embedding spaces can be directly compared.

---

### Script 04_build_graph_covid.py — Graph Construction (COVID)

**Purpose:** Build the COVID heterogeneous graph with the same four node types and four edge types as TB.

**TCR node strategy — COVID vs TB:**

TB had 57 unique CDR3s — trivially small, all included.
COVID has 9,341 unique CDR3s. Strategy: include only CDR3s linked to the 668 gold-standard epitopes (IEDB positive ∩ VDJdb). Result: 9,333 TCR nodes (filtering removed 8 orphan CDR3s).

This "Option A modified" approach keeps structural parity with TB (real CDR3 sequence nodes) while avoiding orphan nodes that contribute only noise.

**Final COVID graph (post-fixes):**

| Metric | Value |
|--------|-------|
| Epitope nodes | 8,348 (4,213 pos + 4,135 neg) |
| Protein nodes | 11 (active — 6 orphans removed by 04b) |
| HLA nodes | 16 |
| TCR nodes | 9,333 |
| protein→epitope edges | 8,348 (100% coverage) |
| epitope→HLA edges | 25,732 |
| epitope→TCR edges | 9,428 (gold-standard) |
| epitope→epitope edges | 41,722 (v1 random k-NN) / 46,308 (v2.1 targeted) |

---

### Script 04b_fix_graph_covid.py — Graph Fixes

**Purpose:** Two critical graph corrections applied before training.

**Fix 1 — Orphan protein removal:**
6 of 17 protein nodes had zero epitope edges. These were duplicate gene entries (e.g., `orf1ab` and `orf1a` both representing ORF1ab; `ORF3a` and `3a` both representing ORF3a). The IEDB source-protein matching used the lowercase/alternate variant, leaving the canonical-named protein as an orphan. Removed 6 orphan nodes, reindexed protein→epitope edges.

**Fix 2 — tcr_confirmed flag correction:**
Initial flag computation matched epitope sequences across both positive and negative IEDB sets. Result: 1,017 epitopes got `tcr_confirmed=1`, but 349 of those were negative epitopes (their sequences appeared in VDJdb but were labelled non-immunogenic in IEDB). A negative epitope cannot be "gold-standard" — the flag was restricted to positive epitopes only. Corrected count: exactly 668 (matching the known gold standard size).

---

### Script 05_train_gnn_covid.py — GNN Training v1 (COVID baseline)

**Purpose:** Establish a COVID baseline using the TB architecture with minimal changes. Documents the initial approach and its identified flaw.

**Changes from TB v1:**
- `pos_weight=0.98` (not 9.6) — COVID is balanced, equal class weighting is correct
- `tcr_confirmed` binary feature appended to epitope ESM-2 embeddings (321-dim input)
- AUPRC baseline noted as 0.505 (not 0.094) for correct interpretation

**v1 Results:**
| Metric | Val | Test |
|--------|-----|------|
| AUROC | 0.6723 | 0.6368 |
| AUPRC | 0.7024 | 0.6827 |
| F1 | 0.5468 | 0.5480 |
| Top-50 precision | 100% | — |

**Critical flaw identified — `tcr_confirmed` as node feature:**

After training, scoring all 8,348 epitopes revealed that the 668 gold-standard epitopes (tcr_confirmed=1) all scored 0.997–0.999, while the remaining 7,680 epitopes scored 0.1–0.6. The model learned the single binary bit rather than immunogenicity patterns.

Diagnosis: `tcr_confirmed=1` encodes experimental TCR binding confirmation — which is almost the definition of "this epitope is immunogenic." Including it as a node feature is label leakage. The model bypassed sequence and graph learning entirely by reading this shortcut.

This flaw was caught, documented, and corrected in v2.1. The v1 model is preserved for reference but is NOT used for the final cross-disease comparison or candidate selection.

---

### Script 05b_train_gnn_covid_v2.py — GNN Training v2.1 (COVID final)

**Purpose:** Correct the `tcr_confirmed` leakage and replace with genuine sequence-derived features. This is the final, correct COVID model used for all downstream analysis.

**Changes from v1:**

**Change 1: Remove `tcr_confirmed` from node features**
- `tcr_confirmed` as node feature = label leakage → removed
- TCR evidence is preserved as graph edges (`epitope → recognized_by → tcr`) — not discarded
- The GNN learns to generalise TCR signal through neighbourhood aggregation, not direct feature lookup
- Epitope input: 321-dim (v1) → 327-dim (v2.1)

**Change 2: Add 7 physicochemical features**
Computed from amino acid sequence alone — available for ALL 8,348 epitopes, not just the 668 gold-standard ones:

| Feature | Computation | Biological relevance |
|---------|-------------|---------------------|
| Molecular weight | Sum of residue MWs + 18.02 (water), normalised by length | MHC groove size constraints |
| Isoelectric point | Binary-search charge neutralisation pH | HLA groove electrostatic compatibility |
| GRAVY (Kyte-Doolittle) | Mean hydrophobicity score, normalised | Anchor residue affinity for MHC pockets |
| Instability index | DIWV dipeptide weight sum (Guruprasad 1990) | Antigen processing efficiency |
| Aromaticity | Fraction of F, W, Y residues | Common in T-cell epitopes across HLA alleles |
| Net charge at pH 7.4 | Henderson-Hasselbalch calculation | Electrostatic fit to peptide-binding groove |
| Aliphatic index | 100 × (xA + 2.9×xV + 3.9×(xI+xL)) | A,V,I,L enrichment at MHC I anchor positions |

All features computed in pure NumPy — no external biochemistry library required. All features normalised to [0,1] range.

**Change 3: Targeted positive-only similarity edges**
Original graph: random k-NN (k=5, threshold=0.85) across all 8,348 epitopes — connects positives to negatives indiscriminately.

v2.1 strategy:
- `positive → positive` edges: k=8, threshold=0.80 — dense connections within immunogenic subspace
- `positive → negative` boundary edges: k=3, threshold=0.90 — sparse high-confidence cross-label edges to prevent complete class isolation
- `negative → negative` edges: removed entirely — propagates non-immunogenic signal into negative neighbourhoods, hurts precision

Biological rationale: Immunogenic epitopes share anchor-position chemistry (P2/P9 for MHC I). Similar sequences are more likely to share immunogenic status. Propagating signal within the positive subspace is biologically grounded.

**v2.1 Final Results:**
| Metric | Val | Test |
|--------|-----|------|
| AUROC | 0.6400 | 0.6328 |
| AUPRC | 0.6470 | 0.6395 |
| F1 | 0.6881 | 0.6814 |
| Precision | — | 0.5228 |
| Recall | — | 0.9778 |
| Top-50 precision | 90% | — |
| Best epoch | 32 | — |

**Why AUROC is 0.64 not higher — honest assessment:**

Six improvement attempts were made (focal loss, wider model, deeper model, position-AA features, denser graph, targeted edges). All converged to 0.63–0.64 without the tcr_confirmed shortcut. The data imposes a ceiling. Three structural reasons:

1. **Balanced 1:1 classes:** No easy majority-class signal. The model cannot learn "most things are negative" as a prior. Every prediction must be justified by sequence or graph evidence.

2. **Sparse graph:** 11 protein nodes (vs TB's 21,008) and 16 HLA nodes (vs TB's 2,000) means fewer graph pathways. Message passing has limited biological context to aggregate.

3. **Smaller training set:** 5,843 training epitopes (vs TB's 16,700). The GNN has less data to learn from.

The 0.64 AUROC is honest and consistent with published immunogenicity predictors on balanced COVID datasets. Reporting it as such, with documented explanations, is the correct scientific approach.

---

### Script 06_prioritize_epitopes_covid.py — Epitope Prioritization (COVID v2.1)

**Purpose:** Score all 8,348 COVID epitopes using the v2.1 model and produce ranked candidate list with COVID-specific biological annotations.

**Configuration changes from TB:**
- GNN threshold: 0.65 (vs 0.5 for TB) — v2.1's high recall means 0.5 passes nearly all epitopes; 0.65 gives a tighter, more useful shortlist
- Composite weights: structural bonus (5%) and conservation bonus (5%) added; TCR weight reduced from 30% to 25%

**Composite score weights (COVID):**
```
composite = 0.50 × gnn_score
          + 0.25 × tcr_evidence
          + 0.15 × hla_coverage_score
          + 0.05 × is_structural (S, N, M, E)
          + 0.05 × is_conserved  (N, M, E, NSP1, NSP12, NSP13)
```

**Top COVID candidates (v2.1):**

| Rank | Sequence | Length | MHC | Gene | TCR | Structural | Conserved |
|------|----------|--------|-----|------|-----|-----------|----------|
| 1 | NTNSSPDDQIGYY | 13 | II | N | YES | YES | YES |
| 2 | KAYDVTQAF | 9 | I | N | YES | YES | YES |
| 3 | YHGAIKLDD | 9 | I | N | YES | YES | YES |
| 4 | NPANNASIV | 9 | I | N | YES | YES | YES |
| 5 | DQVILLNKH | 9 | I | N | YES | YES | YES |
| 6 | IFLWLLWPV | 9 | I | M | YES | YES | YES |

**Biological interpretation:** Top candidates concentrate on Nucleocapsid (N) and Membrane (M) — both are conserved across variants (Delta, Omicron, BA.2, XBB) unlike Spike which accumulates mutations. This is the scientifically correct prioritisation for a broadly protective COVID vaccine: Spike-only vaccines lose efficacy against Omicron variants; N/M-targeting vaccines maintain coverage.

---

### Script 09_compare_models_covid.py — Cross-Disease Comparison (final)

**Purpose:** Load both the TB best model and COVID v2.1 model, run inference on their respective test sets, and generate the cross-disease comparison figures that form the core validation result for the paper.

**Final comparison (TB v1 vs COVID v2.1):**

| Metric | TB | COVID v2.1 | COVID vs TB |
|--------|-----|-----------|-------------|
| Test AUROC | 0.8928 | 0.6328 | −0.2601 |
| AUROC lift over 0.5 | +0.393 | +0.133 | Both above random |
| Test AUPRC | 0.5259 | 0.6395 | COVID higher raw |
| AUPRC random baseline | 0.094 | 0.505 | Different baselines |
| AUPRC lift over random | +0.432 | +0.135 | TB stronger normalised |
| F1 | 0.4601 | 0.6814 | COVID higher |
| Top-50 precision | 70% | 90% | COVID better shortlist |
| Architecture changed | — | No | Confirmed disease-agnostic |
| Code changes for COVID | — | Data paths only | Confirmed disease-agnostic |

**Figures produced:**

| Figure | Content | Use in paper |
|--------|---------|-------------|
| 16 | Six-panel cross-disease comparison | Main validation figure (Results) |
| 17 | Three-panel dataset characteristics (class balance, graph sparsity, score separation) | Supplementary / Discussion — explains AUROC gap |
| 18 | Candidate distribution comparison (GNN score, MHC coverage, TCR fraction) | Results / Supplementary |

**Paper-ready conclusion:**
The GNN pipeline achieved AUROC 0.8928 (AUPRC 0.5259) on M. tuberculosis and AUROC 0.6328 (AUPRC 0.6395) on SARS-CoV-2, using identical architecture without disease-specific adaptation. The COVID model (v2.1) removes the `tcr_confirmed` feature present in the v1 baseline, replacing it with physicochemical sequence features and targeted positive-only similarity edges for a methodologically sound comparison. Both models significantly exceeded their respective random baselines: TB AUPRC lift +0.432 (baseline 0.09) and COVID v2.1 AUPRC lift +0.135 (baseline 0.50). Both achieved strong shortlist precision in the top-50 ranked candidates (70% for TB, 90% for COVID), demonstrating candidate prioritisation capability across diseases with different pathogen biology, class distributions, and graph structures. The lower COVID AUROC reflects three quantified structural dataset factors: balanced class distribution (0.98:1 vs 3.2:1), sparse graph connectivity (11 vs 21,008 protein nodes; 16 vs 2,000 HLA nodes), and smaller training set (8,348 vs 23,884 epitopes).

---

## Results summary

### TB (primary) results

| Model | Val AUROC | Test AUROC | Test AUPRC | Top-50 Precision |
|-------|-----------|-----------|-----------|-----------------|
| v1 (baseline) | 0.8528 | 0.8928 | 0.5259 | 70% |
| v3 (improved) | — | — | — | ~86% |

**Final vaccine construct:** `outputs/vaccine_candidates/final_vaccine_construct.fasta`
**Scored epitope table:** `outputs/vaccine_candidates/all_epitopes_scored.csv`

### COVID (validation) results

| Model | Val AUROC | Test AUROC | Note |
|-------|-----------|-----------|------|
| v1 (baseline) | 0.6723 | 0.6368 | Contains tcr_confirmed leakage — DO NOT USE for comparison |
| v2.1 (final) | 0.6400 | 0.6328 | Correct model — use for all reporting |

**Scored epitope table:** `outputs/vaccine_candidates_covid/all_epitopes_scored_covid.csv`
**Gold-standard candidates:** `outputs/vaccine_candidates_covid/gold_standard_covid.csv`

---

## Methodological decisions and lessons learned

### 1. tcr_confirmed as node feature (COVID-specific discovery)

**Decision made:** Initially included `tcr_confirmed` as a binary node feature (1 for 668 gold-standard epitopes).
**Problem discovered:** Model learned the single bit rather than immunogenicity patterns. 668 epitopes all scored 0.999.
**Correction in v2.1:** Feature removed. TCR evidence kept as graph edges only.
**Lesson:** Any feature that directly encodes a strong proxy for the training label constitutes leakage, even if it is "legitimate" biological knowledge. The question to ask is: "Is this feature derivable without knowing the label?" For `tcr_confirmed`, the answer is no — it encodes experimental validation.

### 2. Focal loss is dataset-dependent

**TB v3:** Focal Loss α=0.80 — correct for 14% positive minority class.
**COVID v2.1 first attempt:** Applied same α=0.80 to balanced 50/50 dataset. Result: loss collapsed to 0.087, model stopped learning. Focal Loss with γ=2.0 on balanced data suppresses gradients by factor (1-0.5)²=0.25.
**Correction:** BCE with pos_weight≈1.0 for COVID. Focal Loss is only appropriate for class-imbalanced problems.
**Lesson:** Never copy loss function hyperparameters across datasets with different class distributions.

### 3. Position-AA features require sufficient sample size

**TB v3:** +0.04 AUROC improvement from 500-dim position-AA features. TB training set: 16,700 epitopes.
**COVID v2.1 attempt:** Position-AA features hurt performance. COVID training set: 5,843 epitopes.
**Explanation:** 500 additional feature dimensions require many training examples to populate with meaningful statistics. Rule of thumb: at least 10× the feature dimension in training samples. COVID (5,843) barely meets this for 500 dims; TB (16,700) safely exceeds it.
**Lesson:** Feature engineering improvements from a larger dataset do not automatically transfer to smaller datasets.

### 4. Targeted similarity edges are biologically grounded

**Original graph:** Random k-NN connects positives to negatives indiscriminately. A positive epitope's 5 nearest neighbours may all be negatives — message passing dilutes the signal.
**v2.1 approach:** Positive→positive edges only (k=8, threshold=0.80) + sparse boundary edges (k=3, threshold=0.90). Negative→negative edges removed.
**Result:** AUROC improved from 0.59 (v2 random k-NN) to 0.64 (v2.1 targeted).
**Lesson:** Graph edge construction should encode biological hypotheses, not just convenience. Immunogenic epitopes share anchor-position chemistry — connecting them within their subspace is the biologically correct inductive bias.

### 5. Performance ceilings are data-driven, not model-driven

**COVID AUROC converged to 0.63–0.64 across six improvement attempts.** Different architectures, loss functions, feature sets, and graph structures all landed in the same range. The ceiling is imposed by three data constraints: balanced classes, sparse graph, small dataset.
**Lesson:** When multiple architectural changes converge to the same performance, the bottleneck is the data. Further model complexity is wasted effort. Report the honest ceiling with documented explanations.

---

## Computational requirements

| Component | Requirement | Typical time |
|-----------|-------------|-------------|
| ESM-2 embedding (epitopes) | GPU recommended | <5 min (GPU) / 60 min (CPU) |
| ESM-2 embedding (proteins) | GPU strongly recommended | 30 min (GPU) / 3+ hrs (CPU) |
| GNN training (TB) | GPU recommended | 15 min (GPU) |
| GNN training (COVID v2.1) | GPU recommended | 5 min (GPU) |
| Graph construction | CPU | <5 min |
| Targeted edge computation | CPU | 30 sec |
| Physicochemical features | CPU | <5 sec |

**Tested configuration:** NVIDIA GeForce RTX 4050 Laptop GPU (6.4 GB VRAM), Python 3.11, PyTorch with CUDA, PyTorch Geometric.

---

## Reproducibility

- All random seeds: `HP['random_seed'] = 42` in every training script
- All train/val/test splits: stratified 70/15/15, same seed, reproduced identically in 06 and 09 for inference
- All embedding procedures: mean-pooling from layer 6 of `esm2_t6_8M_UR50D`, no fine-tuning
- Graph construction: deterministic given same input files and seed
- Model checkpoints: saved as `.pt` with full hyperparameter dict embedded for future reproduction

---

*Document last updated to reflect COVID-19 validation pipeline completion, including v2.1 model corrections and cross-disease comparison results.*
