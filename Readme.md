# OriGENE

**Cancer gene classification using deep convolutional neural networks on epigenetic marker enrichment profiles**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Snakemake](https://img.shields.io/badge/snakemake-≥6.0-brightgreen.svg)](https://snakemake.readthedocs.io)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)

> **Authors:** Marc Pielies-Avelli, Ming-Heng Hsiung, Dr. Alin S. Tomoiaga, Dr. Mattias Ohlsson, Dr. Victor Olariu

---

## Overview

OriGENE is a deep learning framework that predicts cancer driver genes from histone modification enrichment profiles. By training a 1D convolutional neural network on ChIP-seq signals around gene regions, OriGENE primarily distinguishes **cancer driver genes (CD)** from **neutral genes (NG)**, with performance validated against curated cancer gene reference sets.

**Key capabilities:**

- End-to-end Snakemake workflows for reproducible data acquisition and preprocessing
- Multi-track epigenetic input supporting cancer and normal tissue comparisons
- Mutation feature integration for the mutation-aware branch
- Inception-style CNN architecture for cancer gene classification

---

## Architecture Overview

![Simplified conceptual overview of the OriGENE architecture and its extended mutation-aware branch.](assets/figures/origene-architecture-overview.jpg)

*Figure 1. Simplified conceptual overview of the OriGENE architecture and its extended mutation-aware branch.*

---

## Repository Structure

```text
OriGENE/
|- epigenetic_snakemake_pipline/      # ChIP-seq preprocessing workflow
|- mutation(VCF)_snakemake_pipeline/  # VCF generation workflow
|- mutation_feature_pipeline/         # Mutation feature extraction
|- Data/                              # Gene lists, reference files, main-figure inputs
|- analysis_scripts/                  # Gene-disjoint training/evaluation and padding analysis
|- Supplementary/                     # Supplementary result scripts and input documentation
|- assets/                            # Figures and other repository media
|- Tables/                            # Output summary tables
|- OriGENE_main_code.ipynb            # Main model: training, evaluation, visualization
|- OriGENE_Mutation_Branch.ipynb      # Mutation-aware model variant
|- Data-visualization.ipynb           # Main manuscript Figures 1–7
|- Epigenetics_newgenes.py            # Gene-level signal extraction helper
|- envs/                              # Notebook environment
`- Readme.md
```

Directory guides: [Data](Data/Readme.md), [analysis scripts](analysis_scripts/README.md), and [supplementary analyses](Supplementary/README.md).


---

## Biological Background

Post-translational histone modifications reshape chromatin accessibility and regulate transcription. Promoter-proximal H3K4me3 enrichment marks actively transcribed genes, while other histone marks help define active or poised regulatory regions. Aberrant patterns of these marks are a hallmark of oncogenic transformation.

OriGENE exploits the fact that cancer driver genes exhibit systematically different epigenetic landscapes compared to neutral genes, enabling CD-versus-NG classification from enrichment profiles alone.

**Main labels used in the project:**

| Label | Definition |
|---|---|
| CD | Cancer driver |
| NG | Neutral gene (reference negative class) |

OG and TSG remain useful biological background categories in the source annotations, but the main training setup in this repository is centered on CD versus NG labels.

---

## Data sources and accessions

The analyses use publicly available ChIP-seq data and curated gene annotations. The GEO links below point to the **source series**; only the relevant samples and histone-mark tracks were selected for OriGENE, not every sample in each series.

| Source | Accession / reference | Use in OriGENE |
|---|---|---|
| Paired lung and liver normal–tumour ChIP-seq | [GEO GSE67471](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE67471) | Main H3K4me3 analyses |
| ENCODE histone-modification ChIP-seq | [GEO GSE31755](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE31755) | NT2-D1 multi-marker analyses |
| Colorectal cancer ChIP-seq | [GEO GSE156613](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156613) | External colorectal validation |
| Gastric adenocarcinoma chromatin profiling | [GEO GSE51776](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE51776) | External gastric validation |

**Gene labels and genomic annotation**

- **Cancer drivers (CDs):** [COSMIC Cancer Gene Census](https://cancer.sanger.ac.uk/census), **version 96**. The project-specific gene file is [`Data/CD_Binary_All_CGCv.96.csv`](Data/CD_Binary_All_CGCv.96.csv). COSMIC may require registration for full source-data downloads; the current online catalogue may differ from v96.
- **Neutral genes (NGs):** DORGE (Lyu *et al.*, 2020), *Science Advances*, [doi:10.1126/sciadv.aba6784](https://doi.org/10.1126/sciadv.aba6784). The reference NG selection and alternative gene-length-matched selections are described in the manuscript; the full source NG pool should not be confused with the selected 727-gene panel.
- **Cancer-type-specific annotations:** CancerMine (Lever *et al.*, 2019), *Nature Methods*, [doi:10.1038/s41592-019-0422-y](https://doi.org/10.1038/s41592-019-0422-y), used in the tissue-specific analyses where indicated.
- **Gene coordinates and length:** [GENCODE human release 38 (GRCh38.p13)](https://www.gencodegenes.org/human/release_38.html).

The repository also contains project gene-label/coordinate files in [`Data/`](Data/) and [dataset descriptions](Data/Readme.md). GEO series include samples beyond those analyzed in the manuscript; consult the dataset metadata and manuscript for the exact subset.

---

## Pipeline Overview

The full workflow consists of five stages. Each stage can be reproduced independently.

```text
Raw sequencing data -> Alignment and preprocessing -> Gene-level tracks -> CNN training -> Classification
```

---

## Installation

**Dependencies:**

The preprocessing pipelines provide their own environments.  
For the OriGENE model notebooks, use:

```bash
conda env create -f envs/origene-model.yaml
conda activate origene-model
python -m ipykernel install --user --name origene-model --display-name "Python (origene-model)"
```

**Clone the repository:**

```bash
git clone https://github.com/SueH2000/OriGENE.git
cd OriGENE
```

---

## Example dataset: evaluate released predictions

The repository includes small, ready-to-read **example evaluation data** in [`Data/Main_figures/single_track_40kb/`](Data/Main_figures/single_track_40kb/). After completing the [Installation](#installation) steps, run the following from the repository root:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

folder = Path("Data/Main_figures/single_track_40kb")
prefix = "OriGENE_Lung_I_40kbp_singletrack"
y_true = np.load(folder / f"{prefix}_ytrue.npy").ravel()
y_prob = np.load(folder / f"{prefix}_ypred.npy").ravel()
assert y_true.size == y_prob.size, "Labels and predictions must align"
assert set(np.unique(y_true)).issubset({0, 1}), "Expected binary labels"

print(f"Number of evaluated genes: {y_true.size}")
print(f"AUROC: {roc_auc_score(y_true, y_prob):.3f}")
print(f"AUPRC (average precision): {average_precision_score(y_true, y_prob):.3f}")
PY
```

**Input:** two versioned NumPy arrays, `*_ytrue.npy` (binary CD=1 / NG=0 labels) and `*_ypred.npy` (corresponding OriGENE predicted probabilities). **Output:** the gene count, AUROC, and average-precision AUPRC printed to the terminal. Details on saved array formats and gene ordering are in [`Data/Numpy_arrays/Readme.md`](Data/Numpy_arrays/Readme.md).

**Scope:** this checks evaluation using released predictions; it does **not** rerun CNN training or inference from raw ChIP-seq signals. For the input-processing and model-training workflow, see [Workflows](#workflows) and `OriGENE_main_code.ipynb`. Figures can be regenerated using the instructions under [Reproducing the main manuscript figures](#reproducing-the-main-manuscript-figures).

---

## Workflows

### 1. Epigenetic data acquisition (`epigenetic_snakemake_pipline/`)

Downloads or prepares ChIP-seq input, aligns to hg38, and produces sorted bedGraph signal files.

```bash
cd epigenetic_snakemake_pipline
snakemake -s workflow/Snakefile -p --use-conda -j 6
```

**Processing steps (automated):**

| Step | Tool | Output |
|---|---|---|
| Input preparation | SRA toolkit / FASTQ input | `.fastq.gz` |
| Align | `bowtie2` | sorted `.bam` |
| Deduplicate | `samtools rmdup` | deduplicated `.bam` |
| Coverage | `bedtools genomecov` | `.bedGraph` |
| Sort | UCSC / sorting step | `.bedgraph.sorted` |

### 2. Mutation feature pipeline (`mutation_feature_pipeline/`)

Processes VCF files into per-gene mutation-derived features for the mutation-aware branch of OriGENE.

```bash
cd mutation_feature_pipeline
snakemake -s workflow/Snakefile --configfile config/config.yaml --use-conda --cores 4
```

If you need to generate VCF files first, use the upstream workflow in `mutation(VCF)_snakemake_pipeline/`.

### 3. Gene-level track extraction

Split chromosome-level signal files into fixed-width windows around genes using `Epigenetics_newgenes.py` and the metadata in `Data/diagnostic_file.tsv`.

Example:

```python
from Epigenetics_newgenes import create_txt

create_txt(
    Diagnostic_file="Data/diagnostic_file.tsv",
    PATH="path/to/chromosome_files/",
    outPATH="path/to/gene_tracks/",
    shift=2000,
)
```

Output filenames encode metadata such as dataset, sample, marker, gene, strand, and label.

### 4. Preprocessing

Run inside `OriGENE_main_code.ipynb`:

- zero-padding or cropping to uniform length,
- handling missing signal positions,
- reshaping to model input tensors,
- train, validation, and test partitioning.

### 5. Model training and evaluation

Open and run `OriGENE_main_code.ipynb`.  
For the mutation-aware variant, use `OriGENE_Mutation_Branch.ipynb`.

For the 40-kbp paired gene-disjoint training and transfer evaluation used for Figure 7, see [analysis_scripts/README.md](analysis_scripts/README.md). Fixed gene-set and dataset settings are included in `analysis_scripts/train_gene_disjoint.py`.

---

## Model Architecture

OriGENE processes 1D epigenetic signal sequences through three functional blocks:

**Input encoding**

- Fixed-length genomic signal windows
- One or more parallel tracks
- Early pooling to reduce dimensionality

**Feature extraction**

- Stacked 1D convolutional layers
- Inception-style modules for multi-scale feature extraction
- Repeated pooling for progressive compression

**Classification head**

- Flatten layer
- Fully connected layers
- Final sigmoid output for cancer-driver prediction

---

## Key References

- Jie Lyu et al. (2020). *Genome-wide identification of cancer drivers.* Science Advances. https://doi.org/10.1126/sciadv.aba6784
- Cancer Gene Census: https://cancer.sanger.ac.uk/census
- UCSC Genome Browser: https://genome.ucsc.edu/

---

## Glossary

| Term | Definition |
|---|---|
| TSS | Transcription start site |
| CD | Cancer driver |
| NG | Neutral gene (reference negative class) |
| ChIP-seq | Chromatin immunoprecipitation sequencing |
| bedGraph | Genome coverage format |

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Citation

If you use OriGENE in your research, please cite the associated thesis, paper, or software record accompanying this repository.

## Reproducing the main manuscript figures

Run `Data-visualization.ipynb` from the repository root, with `Data/Main_figures/` in place. Run cells in order. The notebook produces eight PNG files for seven manuscript figures (Figure 3 has separate panels a and b).


## Archived software

Version v1.1.0 is cited in the manuscript as https://doi.org/10.5281/zenodo.23781902. The development branch contains later changes. The v1.1.0 citation identifies that release, not the current development branch.
