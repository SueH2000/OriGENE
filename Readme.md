# OriGENE

**Cancer gene classification using deep convolutional neural networks on epigenetic marker enrichment profiles**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Snakemake](https://img.shields.io/badge/snakemake-≥6.0-brightgreen.svg)](https://snakemake.readthedocs.io)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

> **Authors:** Marc Pielies-Avellí · Ming-Heng Hsiung · Dr. Alin S. Tomoiaga · Dr. Mattias Ohlsson · Dr. Victor Olariu

---

## Overview

OriGENE is a deep learning framework that predicts cancer driver genes from histone modification enrichment profiles. By training a 1D convolutional neural network on ChIP-seq signals (H3K4me3, H3K4me1, H3K27ac) around transcription start sites, OriGENE classifies genes as **oncogenes (OG)**, **tumor suppressor genes (TSG)**, or **normal genes (NG)** — with performance validated against the Cancer Gene Census (CGC) gold standard.

**Key capabilities:**
- End-to-end Snakemake workflows for reproducible data acquisition and preprocessing
- Multi-track epigenetic input supporting paired cancer/healthy tissue comparisons
- Mutation (VCF) integration branch for somatic variant-aware classification
- Inception-module CNN architecture with interpretable feature maps

---

## Repository Structure

```
OriGENE_published/
├── epigenetic_snakemake_pipeline/   # ChIP-seq download → bedGraph workflow
├── mutation(VCF)_snakemake_pipeline/# VCF processing → VEP
├── mutation_feature_pipeline/       # Mutation feature extraction
├── Data/                            # Gene lists, CGC reference, diagnostic files
├── Tables/                          # Output summary tables
├── OriGENE_main_code.ipynb          # Main model: training, evaluation, visualization
├── OriGENE_mutation_branch.ipynb    # Mutation-aware model variant
├── Data-visualization.ipynb         # Exploratory analysis and figures
├── envs/                            # Environment
└── README.md
```

---

## Biological Background

Post-translational histone modifications reshape chromatin accessibility and regulate transcription. Promoter-proximal H3K4me3 enrichment marks actively transcribed genes, while H3K4me1 and H3K27ac mark active and poised enhancers. Aberrant patterns of these marks are a hallmark of oncogenic transformation.

OriGENE exploits the fact that cancer driver genes exhibit systematically different epigenetic landscapes compared to neutral genes — even across tissue types — enabling cross-dataset generalisation without requiring matched mutations.

**Labels (from CGC v95):**
| Label | Definition |
|---|---|
| OG | Oncogene — promotes tumour growth when activated |
| TSG | Tumour suppressor gene — growth-suppressive when inactivated |
| CD | Cancer driver (OG or TSG combined) |
| NG | Normal gene (non-driver) |

---

## Pipeline Overview

The full workflow consists of five stages. Each stage is independently reproducible via Snakemake.

```
Raw SRA data  →  Alignment (hg38)  →  Gene-level tracks  →  CNN training  →  Classification
```

*(See the diagram rendered in the companion widget or the figure below.)*

---

## Installation

**Dependencies:**

The preprocessing pipelines provide their own environments.  
For the OriGENE model notebooks, use:

```bash
conda env create -f envs/origene-model.yaml
conda activate origene-model
python -m ipykernel install --user --name origene-model --display-name "Python (origene-model)"

**Clone the repository:**

```bash
git clone https://github.com/<org>/OriGENE.git
cd OriGENE
```

---

## Workflows

### 1. Epigenetic data acquisition (`epigenetic_snakemake_pipeline/`)

Downloads ChIP-seq SRA files, converts to FASTQ, aligns to hg38, and produces sorted bedGraph files.

```bash
cd epigenetic_snakemake_pipeline
snakemake --cores 8 --config sra_list=config/sra_accessions.txt
```

**Processing steps (automated):**

| Step | Tool | Output |
|---|---|---|
| Download | `wget` / SRA toolkit | `.sra` |
| Convert | `fastq-dump` | `.fastq` |
| Align | `bowtie2` (hg38 index) | `.sam` |
| Convert + sort | `samtools` | `.bam.sorted` |
| Deduplicate | `samtools rmdup` | `.dedup.bam` |
| Coverage | `bedtools genomecov` | `.bedGraph` |
| Sort | `bedSort` | `.bedGraph.sorted` |

The bedGraph format produced:
```
chr1  10000  10007  1
chr1  10007  10029  2
chr1  10029  10031  4
```

**Data sources:**
- Chen K et al. 2015 (GSE67471) — H3K4me3 in cancer vs normal pairs: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE67471
- ENCODE NT2-D1 (GSE31755) — multi-marker reference: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE31755
- Li Q et al. 2021 (GSE156613) — enhancer landscape in colorectal cancer subgroups (H3K27ac/H3K4me3 ChIP-seq, tumor vs native tissues): https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156613

### 2. Mutation feature pipeline (`mutation_feature_pipeline/`)

Processes somatic VCF files to extract per-gene mutation burden and variant-level features for the mutation-aware model branch.

```bash
cd mutation_feature_pipeline
snakemake --cores 4 --config vcf_dir=path/to/vcfs genome=hg38
```

### 3. Gene-level track extraction

Split bedGraph by chromosome, then extract fixed-width windows around each gene's TSS:

```python
# In OriGENE-code.ipynb or as a standalone script
from Epigenetics_newgenes import create_txt

create_txt(
    diagnostic_file = "Data/gene_diagnostics.csv",
    PATH            = "path/to/chromosome_files/",
    outPATH         = "path/to/gene_tracks/",
    shift           = 2000   # bp upstream of TSS
)
```

Output filenames encode all metadata:
```
{Dataset}_{Sample}_Shifted_{shift}_{marker}_{gene}_{strand}_{label}.txt
# e.g. GSE67471_GSM1647620_Shifted_2000_H3K4me3_SGK1_neg_OG.txt
```

### 4. Preprocessing

Run inside `OriGENE-code.ipynb` (Section: *Data Preprocessing*):

- Zero-padding / cropping to uniform length (5,000 bp or 40,000 bp)
- Gap-filling: missing bins assigned the preceding signal value
- 2 bp flanking zeros at gene boundaries
- Reshape to `(n_samples, bp_length, n_tracks)` for CNN input
- Stratified train / validation / test split

### 5. Model training and evaluation

Open and run `OriGENE-code.ipynb`. All hyperparameters are set in the *Configuration* cell at the top.

For the mutation-aware variant, use `OriGENE_Mutation_Branch.ipynb`.

---

## Model Architecture

OriGENE processes aligned 1D epigenetic sequences through three functional blocks:

**Input encoding**
- Sequences of length 5,000 or 40,000 bp, starting 2,000 bp upstream of TSS
- Multiple tracks in parallel: cancer tissue signal, matched normal tissue signal, additional histone marks
- Initial average pooling reduces dimensionality while preserving peak structure

**Feature extraction**
- Stacked 1D convolutional layers (ReLU) with inception modules for multi-scale filter banks
- Alternating average and max pooling for progressive dimensionality reduction

**Classification head**
- Flatten → fully connected layers (MLP, ReLU)
- Single sigmoid output: 1 = cancer driver (OG/TSG), 0 = normal gene

```
Input tracks (n_tracks × L bp)
        │
   [Avg Pool]          ← dimensionality reduction
        │
   [Conv1D + ReLU]
   [Inception block]   ← multi-scale feature extraction
   [Pool]
        │  (repeated)
   [Flatten]
   [Dense + ReLU]      ← MLP classifier
   [Dense + Sigmoid]
        │
   Output: P(cancer driver)
```

---

## Key References

- Jie Lyu et al. (2020). *Genome-wide identification of cancer drivers.* Science Advances. https://doi.org/10.1126/sciadv.aba6784
- Chen K et al. (2015). GSE67471. https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE67471
- Sherman M.A. et al. (2022). *Genomic maps of chromatin accessibility.* Nature Biotechnology. https://doi.org/10.1038/s41587-022-01353-8
- Cancer Gene Census: https://cancer.sanger.ac.uk/census
- UCSC Genome Browser: https://genome.ucsc.edu/

---

## Glossary

| Term | Definition |
|---|---|
| TSS | Transcription start site |
| OG | Oncogene |
| TSG | Tumour suppressor gene |
| CD | Cancer driver (OG ∪ TSG) |
| NG | Normal (non-driver) gene |
| CGC | Cancer Gene Census |
| ChIP-seq | Chromatin immunoprecipitation sequencing |
| bedGraph | Genome coverage format (chrom, start, end, score) |

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Citation

If you use OriGENE in your research, please cite:

```bibtex
@software{origene2025,
  author  = {Pielies-Avell\'{i}, Marc and Hsiung, Ming-Heng and Tomoiaga, Alin S. and Ohlsson, Mattias and Olariu, Victor},
  title   = {{OriGENE}: Cancer gene classification from epigenetic enrichment profiles},
  year    = {2025},
  url     = {https://github.com/<org>/OriGENE}
}
```
