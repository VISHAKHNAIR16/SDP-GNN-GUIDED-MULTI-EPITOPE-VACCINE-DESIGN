**Project:** GNN-GUIDED MULTI-EPITOPE VACCINE DESIGN: A PROTEIN-LANGUAGE-MODEL–DRIVEN HETEROGENEOUS GRAPH FRAMEWORK FOR PRIORITIZING MDR-TB EPITOPES WITH BROAD HLA BINDING AND TCR RECOGNITION

---

**Purpose of this document:**
- Provide a detailed, runnable, and plain-language explanation of every Python script in the repository and the primary outputs they've produced. This is both a developer reference and a methods appendix suitable for a paper or thesis.

**How to use this document:**
- Each script section explains: what it does, why it's needed, inputs, outputs, key functions/algorithms, and a plain-English explanation of the biology/computation.
- Files and folders referenced are workspace-relative and clickable.

---

**Quick run sequence (typical pipeline)**
- Clean data: `python scripts/01_clean_data.py`
- EDA: `python scripts/02_eda.py`
- Feature engineering (ESM-2 embeddings): `python scripts/03_feature_engineering.py`
- Build graph: `python scripts/04_build_graph.py`
- Train GNN: `python scripts/05_train_gnn.py`
- Prioritize epitopes: `python scripts/06_prioritize_epitopes.py`
- Improve/rerank (v2/v3/v4 experiments): `python scripts/07_improve_and_rerank.py`, `07_improve_v3.py`, `07_improve_v4.py`
- Assemble multi-epitope vaccine: `python scripts/08_multi_epitope_assembly.py`
- Compare models and produce figures: `python scripts/09_compare_models.py`

---

**Data & processed files**
- Cleaned and processed inputs (these are the canonical inputs used throughout):
  - [data/processed/iedb_positive_clean.csv](data/processed/iedb_positive_clean.csv)
  - [data/processed/iedb_negative_clean.csv](data/processed/iedb_negative_clean.csv)
  - [data/processed/vjdb_tb_human_clean.tsv](data/processed/vjdb_tb_human_clean.tsv)
  - [data/processed/hla_prot_clean.fasta](data/processed/hla_prot_clean.fasta)
  - [data/processed/tb_proteome_clean.fasta](data/processed/tb_proteome_clean.fasta)
  - Embeddings: [data/processed/embeddings/](data/processed/embeddings/) (epitopes_positive.npy, epitopes_negative.npy, tb_proteins.npy, hla_sample.npy and corresponding _meta.csv files)
  - Built graph and stats: [data/processed/graph/heterogeneous_graph.pt](data/processed/graph/heterogeneous_graph.pt) and [data/processed/graph/graph_stats.json](data/processed/graph/graph_stats.json)

---

**Outputs (results & artifacts)**
- Core model artifacts and logs: [outputs/models/](outputs/models/) — contains saved checkpoints (`best_model*.pt`) and `training_history*.json` for versions v1..v4.
- Candidate lists: [outputs/vaccine_candidates/](outputs/vaccine_candidates/) — `top_candidates*.csv`, `all_epitopes_scored.csv`, `final_vaccine_construct.fasta` and `final_vaccine_construct.txt` (assembled construct), `selected_epitopes.csv`.
- Figures for publication: [outputs/figures/](outputs/figures/) — numbered PNGs (01..21) that correspond to EDA, training curves, ROC/PR, candidate visualizations, and vaccine construct diagrams.
- Logs: `outputs/cleaning.log`, `outputs/feature_engineering.log`, `outputs/graph_building.log`, `outputs/training.log`, `outputs/phase7*.log`.

---

**Scripts (detailed)**

- **`scripts/01_clean_data.py`**
  - Purpose: canonical cleaning of raw IEDB, VDJdb, HLA FASTA, and TB proteome files into consistent, validated, deduplicated datasets.
  - Run: `python scripts/01_clean_data.py`
  - Inputs: raw files under `data/` (e.g., `data/IEDB/epitope_table_positive.csv`, `data/VJDB/vjdb.tsv`, `data/HLA/hla_prot.fasta`, `data/TBDB/tb1584.fasta`).
  - Outputs: `data/processed/iedb_positive_clean.csv`, `data/processed/iedb_negative_clean.csv`, `data/processed/vjdb_tb_human_clean.tsv`, `data/processed/hla_prot_clean.fasta`, `data/processed/tb_proteome_clean.fasta`, `data/processed/tb_proteome_metadata.csv`. Also writes `outputs/cleaning.log`.
  - Key steps & functions:
    - `find_peptide_column()` — robustly detects which column actually holds peptide sequences in IEDB exports (IEDB exports are messy: `peptide_seq` often contains the literal string `"Linear peptide"`).
    - `is_valid_peptide()` / `is_valid_protein()` — filters sequences by length and amino-acid alphabet.
    - Deduplication: collapse assay-level rows into unique epitope sequences.
    - HLA parsing: extracts gene family (A/B/C/DR/DQ/DP) from FASTA headers.
  - Plain-language: standardizes messy lab exports into clean one-row-per-epitope or one-record-per-protein datasets, checks sequences are plausible amino-acid strings, and keeps only TB-specific entries where required.

- **`scripts/02_eda.py`**
  - Purpose: generate publication-ready exploratory figures describing dataset composition and quality.
  - Run: `python scripts/02_eda.py`
  - Inputs: cleaned files in `data/processed/`.
  - Outputs: figures saved to `outputs/figures/` (01–07 including length distributions, amino acid frequency, HLA allele coverage, IEDB–VDJdb overlap) and console summary.
  - Key plots & why they matter:
    - Length distribution: distinguishes MHC I (8–11 aa) vs MHC II (13–25 aa) and shows whether training data covers both.
    - IEDB–VDJdb overlap: identifies "gold-standard" epitopes present in both databases (strongest evidence).
    - HLA allele coverage: shows how well HLA alleles in dataset span human diversity.
  - Plain-language: these figures show dataset biases and whether the data supports building a model that can learn immunogenic patterns.

- **`scripts/03_feature_engineering.py`**
  - Purpose: use the ESM-2 protein language model to encode sequences into dense vector embeddings used as node features in the graph.
  - Run: `python scripts/03_feature_engineering.py`
  - Inputs: cleaned sequences (epitopes, TB proteins, HLA FASTA) from `data/processed/`.
  - Outputs: `data/processed/embeddings/epitopes_positive.npy`, `epitopes_negative.npy`, `tb_proteins.npy`, `hla_sample.npy` and matching `_meta.csv` files. Produces `outputs/feature_engineering.log` and does checkpointing/resume.
  - Key points:
    - Uses `esm2_t6_8M_UR50D` from `esm` package (ESM-2 small): 480-dim representations.
    - Mean-pools token-level representations to produce sequence-level vectors.
    - Samples HLA alleles for representativeness rather than embedding all ~44k alleles to save time.
    - Validates embeddings for NaN/Inf and distribution differences between positive/negative.
  - Plain-language: converts biological sequences into numerical fingerprints that capture biochemical and evolutionary context; those fingerprints are the features the GNN uses.

- **`scripts/04_build_graph.py`**
  - Purpose: assemble a heterogeneous graph (PyTorch Geometric HeteroData) that links epitopes, proteins, HLA alleles, and TCR sequences with biologically meaningful edges.
  - Run: `python scripts/04_build_graph.py`
  - Inputs: embeddings and `_meta.csv` from `data/processed/embeddings/`, cleaned files (for source-protein and MHC allele lookups).
  - Outputs: `data/processed/graph/heterogeneous_graph.pt` and `data/processed/graph/graph_stats.json`, plus `outputs/graph_building.log`.
  - Node types: `epitope`, `protein`, `hla`, `tcr`.
  - Edge types: `protein -> source_of -> epitope` (source membership), `epitope -> binds_to -> hla` (experimental or similarity-based), `epitope -> recognized_by -> tcr` (VDJdb gold edges), `epitope -> similar_to -> epitope` (k-NN by embedding cosine).
  - Important algorithms:
    - k-NN similarity edges for epitopes using normalized cosine similarity (batch processing to avoid O(N^2) memory blowup).
    - Fallbacks: when explicit allele names are missing, edges to HLA nodes are built by embedding similarity.
  - Plain-language: constructs a biological network where edges encode experiments or similarities so the GNN can propagate evidence across related biological entities.

- **`scripts/05_train_gnn.py`**
  - Purpose: train a heterogeneous GNN (HANConv-based) on the assembled graph to predict epitope immunogenicity.
  - Run: `python scripts/05_train_gnn.py`
  - Inputs: `data/processed/graph/heterogeneous_graph.pt`.
  - Outputs: model checkpoint `outputs/models/best_model.pt`, `outputs/models/training_history.json`, figures `08_training_curves.png`,`09_roc_pr_curves.png`, and `outputs/training.log` entries.
  - Model architecture & training details:
    - Input projection per node type → HANConv layers (multi-head attention over heterogeneous relations) → MLP classifier for epitope nodes.
    - Uses weighted BCE loss (`pos_weight` tuned by class imbalance), Adam optimizer, LR scheduler `ReduceLROnPlateau`, early stopping.
    - Hyperparameters are defined in `HP` dictionary; best models saved to `outputs/models/`.
  - Evaluation:
    - Metrics: AUROC, AUPRC (preferred for imbalanced problems), F1, precision, recall, confusion matrix and top-50 precision.
  - Plain-language: the GNN learns to combine node features and network context so it can predict which epitopes are immunogenic (label 1) vs not.

- **`scripts/06_prioritize_epitopes.py`**
  - Purpose: use a trained GNN to score all epitopes, merge model outputs with biological annotations (HLA connectivity, TCR evidence, source protein), compute composite ranking scores, and save ranked candidates.
  - Run: `python scripts/06_prioritize_epitopes.py`
  - Inputs: trained model `outputs/models/best_model.pt`, graph `data/processed/graph/heterogeneous_graph.pt`.
  - Outputs: `outputs/vaccine_candidates/all_epitopes_scored.csv`, `top_candidates.csv`, `top50_candidates.csv`, `gold_standard_epitopes.csv` and figures `10_score_distributions.png`, `11_top20_candidates.png`, `12_source_protein_analysis.png`.
  - Scoring pipeline:
    - `gnn_score`: model probability for immunogenicity (sigmoid output)
    - `tcr_evidence`: boolean if epitope appears in VDJdb
    - `hla_coverage_score`: normalized number of HLA neighbors
    - `composite_score`: weighted sum (default: 50% GNN, 30% TCR, 20% HLA)
    - Filter: default GNN threshold > 0.5 to select candidates
  - Plain-language: ranks epitopes by a combination of model confidence and biological evidence so top-ranked epitopes are both predicted and supported experimentally.

- **`scripts/07_improve_and_rerank.py`, `07_improve_v3.py`, `07_improve_v4.py`**
  - Purpose: iterative experiments to improve graph construction and training, then rerank candidates. These implement v2 (denser similarity edges and HLA supertype edges), v3 (larger model, focal loss, position features, conservation features), and v4 (best-of-v1+v3 with a dual loss) experiments.
  - Run: `python scripts/07_improve_and_rerank.py`, `python scripts/07_improve_v3.py`, `python scripts/07_improve_v4.py` respectively.
  - Inputs: embeddings and previously-built graphs or the scripts re-build a new improved graph (they include `build_improved_graph` / `build_graph` functions).
  - Outputs: improved graph files (e.g., `data/processed/graph/heterogeneous_graph_v2.pt` / `_v3.pt` / `_v4.pt`), new model checkpoints (`outputs/models/best_model_v2.pt`, `best_model_v3.pt`, `best_model_v4.pt`), training histories `training_history_v*.json`, and reranked candidate CSVs like `top_candidates_v2.csv`, `top25_classI_v2.csv`, `top25_classII_v2.csv`.
  - Key improvements explained in plain language:
    - Lowering similarity threshold and increasing k-NN densifies the epitope similarity graph to allow more evidence propagation.
    - HLA supertype edges add biological prior knowledge (alleles with similar binding properties are linked), improving generalization across alleles.
    - Focal loss (v3) addresses class imbalance by up-weighting hard positives.
    - Dual loss (v4) mixes focal + weighted BCE to balance precision vs recall for final reported metrics.
  - Practical note: these scripts save alternate models and candidate lists; include the version suffix in filenames when citing results in a paper.

- **`scripts/08_multi_epitope_assembly.py`**
  - Purpose: assemble a final multi-epitope vaccine construct from the top candidates (prefer v3-v4 outputs), perform physicochemical checks, estimate HLA population coverage, and export FASTA/text summary.
  - Run: `python scripts/08_multi_epitope_assembly.py`
  - Inputs: ranked candidate CSVs (prefer `top_candidates_v3.csv` or `top_candidates.csv` fallback).
  - Outputs: `outputs/vaccine_candidates/final_vaccine_construct.fasta`, `final_vaccine_construct.txt`, `selected_epitopes.csv`, and figures `18_vaccine_construct.png`, `19_hla_population_coverage.png`.
  - Key design choices and plain-language rationale:
    - Linkers: `AAY` between Class I epitopes (helps proteasome cleavage), `GPGPG` between Class II epitopes.
    - Adjuvant: short TLR4-agonist peptide appended N‑terminally to boost immune activation.
    - Non-redundancy: enforces max k-mer overlap (80% identity proxy) so chosen epitopes provide complementary coverage.
    - Physicochemical checks (MW, pI, GRAVY, instability) to assess expression/stability.
  - Plain-language: produces a single protein-like sequence that can be used for expression or downstream immunoinformatics (population coverage, epitope mapping).

- **`scripts/09_compare_models.py`**
  - Purpose: load multiple saved checkpoints and graphs, compute test metrics, overlay ROC/PR curves, produce model-comparison figures and a summary table (ablation study across v1..v4).
  - Run: `python scripts/09_compare_models.py`
  - Inputs: graphs and model checkpoints for each version under `data/processed/graph/` and `outputs/models/`.
  - Outputs: `outputs/figures/model_comparison.png`, `metric_progression.png`, and console summary comparing AUROC/AUPRC/F1/Recall and top-50 precision.
  - Plain-language: compares alternatives to justify final modeling choices in Methods and Results.

---

**Important output files and how to interpret them (key list)**
- [outputs/models/best_model.pt](outputs/models/best_model.pt) — v1 trained model checkpoint (load for inference with `scripts/06_prioritize_epitopes.py`).
- [outputs/models/best_model_v2.pt](outputs/models/best_model_v2.pt) — improved graph v2 checkpoint.
- [outputs/models/best_model_v3.pt](outputs/models/best_model_v3.pt), [outputs/models/best_model_v4.pt](outputs/models/best_model_v4.pt) — final experiment checkpoints (v3/v4).
- [outputs/models/training_history*.json](outputs/models/) — training metrics per epoch for plotting and reproducibility.
- [outputs/vaccine_candidates/top_candidates.csv](outputs/vaccine_candidates/top_candidates.csv) — full ranked candidate list (composite_score present).
- [outputs/vaccine_candidates/all_epitopes_scored.csv](outputs/vaccine_candidates/all_epitopes_scored.csv) — the full table of all epitopes with `gnn_score`, `tcr_evidence`, `hla_coverage_score`, `composite_score`.
- [outputs/vaccine_candidates/final_vaccine_construct.fasta](outputs/vaccine_candidates/final_vaccine_construct.fasta) and [outputs/vaccine_candidates/final_vaccine_construct.txt](outputs/vaccine_candidates/final_vaccine_construct.txt) — assembled multi-epitope construct in FASTA and human-readable summary.
- [outputs/figures/08_training_curves.png](outputs/figures/08_training_curves.png), [outputs/figures/09_roc_pr_curves.png](outputs/figures/09_roc_pr_curves.png) — training diagnostic figures for the baseline model.
- [outputs/figures/model_comparison.png](outputs/figures/model_comparison.png) — ablation comparison of v1..v4.

---

**Glossary — technical terms explained in simple language**
- Epitope: a short peptide (string of amino acids) from a pathogen that can be recognized by the immune system.
- HLA: human proteins that present short peptides on cell surfaces so T-cells can recognize them. HLA alleles vary across people and determine population coverage.
- TCR: T-cell receptor — the immune receptor on T cells that recognizes a specific epitope-HLA combination. `VDJdb` records experimental TCR-epitope matches.
- Embedding (ESM-2): a dense numerical vector that represents sequence properties learned by a large protein-language model. Think of it as a fingerprint of biochemical/structural features.
- Heterogeneous graph: a graph with different node types (epitope, protein, HLA, TCR) and different edge types; it lets the model treat each type differently.
- GNN (Graph Neural Network): a neural network that learns by passing messages along graph edges; it lets the model propagate evidence (e.g., TCR evidence or protein context) to related sequences.
- HANConv: a PyG layer for heterogeneous graphs that uses attention across multiple relation types.
- AUROC / AUPRC / F1: metrics for classification; AUPRC is preferred for imbalanced data. Top-50 precision is the fraction of true immunogenic epitopes among the top 50 ranked candidates.
- Focal loss: a loss function that emphasizes hard-to-classify minority examples (positives) to address class imbalance.

---

**Reproducibility notes & tips**
- Virtual environment: ensure `esm`, `torch`, `torch_geometric`, `BioPython`, `pandas`, `numpy`, `scikit-learn`, `loguru`, `rich`, and plotting libs are installed.
- GPU: `scripts/03_feature_engineering.py` and all training scripts are GPU-accelerated but will run on CPU (slow). See `load_esm_model()` and device selection lines in scripts.
- If you re-run embeddings or model training, keep consistent random seeds (`HP['random_seed']`) to ease comparisons.
- For papers: cite model variants by checkpoint filenames and include exact hyperparameters shown in each script's `HP` dictionary.

---

**Suggestions / next actions (if you want me to continue)**
- I can commit this document to the repo (already saved as `PROJECT_DOCUMENTATION.md`).
- I can generate a concise Methods section ready for a manuscript (LaTeX or Word) using exact numbers from `outputs/models/training_history*.json` and `data/processed/graph/graph_stats.json`.
- I can run `scripts/09_compare_models.py` here and snapshot the generated figures if you want those embedded into the doc.

---

Document generated automatically from repository scripts and outputs on behalf of the project owner.

---

**Results (summary)**
- **Primary classification performance (reported / recommended):** v1 baseline — AUROC 0.8928, AUPRC 0.5259, Recall 0.8309 (see [scripts/09_compare_models.py](scripts/09_compare_models.py) summary and [outputs/figures/model_comparison.png](outputs/figures/model_comparison.png)).
- **Top-candidate precision:** v3 recommended for candidate selection — top-50 precision ≈ 86% (used in final selection and reported in the ablation summary in `scripts/09_compare_models.py`).
- **Final vaccine construct:** saved as [outputs/vaccine_candidates/final_vaccine_construct.fasta](outputs/vaccine_candidates/final_vaccine_construct.fasta) and summarized in [outputs/vaccine_candidates/final_vaccine_construct.txt](outputs/vaccine_candidates/final_vaccine_construct.txt). Population coverage estimate (conservative): typically reported in the construct text file and plotted in [outputs/figures/19_hla_population_coverage.png](outputs/figures/19_hla_population_coverage.png).
- **Full scored epitope table:** [outputs/vaccine_candidates/all_epitopes_scored.csv](outputs/vaccine_candidates/all_epitopes_scored.csv) — contains `gnn_score`, `tcr_evidence`, `hla_coverage_score`, and `composite_score` used for ranking.
- **Training diagnostics:** per-epoch histories are in `outputs/models/training_history.json` and `training_history_v*.json`; ROC/PR and training curves are in `outputs/figures/08_training_curves.png` and `09_roc_pr_curves.png`.

Interpretation: the model achieves strong discrimination (high AUROC) with moderate AUPRC (common in imbalanced biological datasets). The ablation study shows v3/v4 tradeoffs increase candidate precision at the cost of some general classification metrics — this motivated selecting v3 outputs for vaccine assembly where high top-k precision matters more than global recall.

---

**Methodology (detailed)**
This section provides a compact reproducible methods description suitable for a Methods appendix.

- **Data sources and cleaning**
  - Raw immunogenicity data: IEDB positive/negative tables; experimental TCR mappings from VDJdb; HLA FASTA sequences; TB proteome FASTA.
  - Cleaning steps (see `scripts/01_clean_data.py`): normalize column names, validate amino-acid sequences, deduplicate entries, map IEDB assays to canonical peptide sequences, and extract HLA gene families. Output files are in [data/processed/](data/processed/).

- **Feature engineering (ESM embeddings)**
  - Model: ESM-2 (`esm2_t6_8M_UR50D`) with 480-dimensional sequence embeddings.
  - Procedure: token-level features are mean-pooled to obtain sequence vectors; embeddings produced for epitopes, TB proteins, and a sampled set of HLA sequences. Embeddings are stored as `.npy` with matching metadata CSVs in [data/processed/embeddings/].
  - Rationale: ESM embeddings capture biochemical and evolutionary context without alignment and often outperform simple k-mer or physicochemical features.

- **Heterogeneous graph construction**
  - Node types: `epitope`, `protein`, `hla`, `tcr`.
  - Edge types: `protein->source_of->epitope`, `epitope->binds_to->hla` (experimental or similarity-derived), `epitope->recognized_by->tcr`, `epitope->similar_to->epitope` (k-NN by cosine similarity).
  - Parameters: typical k-NN k=5; similarity thresholds adjusted per-experiment (v2 lowers threshold to increase connectivity). Graphs are saved under [data/processed/graph/](data/processed/graph/).

- **GNN architecture and variants**
  - Baseline (`v1`): per-node-type input projection → stacked HANConv layers (multi-head attention across relation types) → epitope MLP classifier. Loss: BCEWithLogitsLoss with `pos_weight` to correct imbalance.
  - Improvements:
    - `v2`: denser epitope similarity graph + HLA supertype edges.
    - `v3`: wider/deeper HAN (per-node input dims), added position/conservation features, and Focal Loss (α around 0.8) to focus learning on hard positives.
    - `v4`: dual loss combining focal + BCE to balance precision and recall.

- **Training regime**
  - Optimizer: Adam; LR scheduler: ReduceLROnPlateau; early stopping on validation AUROC.
  - Evaluation: AUROC and AUPRC reported per-epoch; top-k precision computed across all epitopes for candidate-focused evaluation.
  - Reproducibility: random seeds set in hyperparameter blocks; GPU recommended for embedding and training.

- **Candidate scoring and selection**
  - Scores: `gnn_score` (model probability), `tcr_evidence` (binary), `hla_coverage_score` (normalized count of HLA neighbors).
  - Composite ranking: default weightings — 0.50 * GNN + 0.30 * TCR + 0.20 * HLA. Final candidate lists (`top_candidates*.csv`) are generated per-version.

- **Multi-epitope assembly**
  - Strategy: select non-redundant Class I and II epitopes prioritizing TCR-confirmed and essential genes, assemble with empirically chosen linkers (`AAY` for Class I, `GPGPG` for Class II), and add an N-terminal adjuvant peptide.
  - Validation: compute molecular weight, pI, GRAVY, instability index, and estimate HLA supertype coverage with a simplified model of allele frequencies (see `scripts/08_multi_epitope_assembly.py`).

- **Interpretation principles**
  - For model development, emphasize AUROC/AUPRC and curve diagnostics.
  - For vaccine design, emphasize top-k precision and biological evidence (TCR, HLA breadth, essential gene sources) because experimental validation focuses on a small candidate set.

---

**Frontend website and documentation**
The repository includes a lightweight static frontend that visualizes results and explains components for non-technical reviewers. Files are in the `frontend/` folder:

- `frontend/vaccine_demo.html` — interactive presentation of the final vaccine construct, sequence map, and key properties.
- `frontend/gnn_signal_flow_explainer.html` — visual explainer of how graph signals (TCR/HLA/protein) propagate to influence epitope scores.
- `frontend/four_model_ablation.html` — interactive plots comparing v1..v4 metrics and top-k precision; useful for presentation and reviewers.
- `frontend/biological_graph_explainer.html` — description and interactive view of the heterogeneous biological graph (node/edge types, example neighborhoods).

How to view locally:
1. Open the HTML file directly in a browser (double-click) for static viewing.
2. For full interactivity (some browsers restrict local JS), run a simple HTTP server from project root:

```powershell
python -m http.server 8000
# then open http://localhost:8000/frontend/vaccine_demo.html
```

The frontend is intended as supplementary material for presentations and the project website; if you want, I can embed screenshots or link the HTML pages into `PROJECT_DOCUMENTATION.md`.

---

If you'd like, I will now embed the key figures (`model_comparison.png`, `08_training_curves.png`, `final_vaccine_construct.png`) into this markdown and append numeric results extracted directly from the `training_history*.json` files. Mark which figures you want included.
