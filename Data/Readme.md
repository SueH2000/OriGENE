# Data

This folder contains the main reference files and derived outputs used in OriGENE.

## What is in this folder

### 1. Dataset information and performance metrics

The subfolder `Dataset information and Performance metrics/` contains supporting information about the datasets used in the project, including accession-related details and summary performance tables.

Use these files when you want to understand:

- which public datasets were used,
- which histone marks or study settings were included,
- how OriGENE performed in different evaluation scenarios.

### 2. Gene label files

The following files in this folder define gene sets and labels used in the study:

- `CD_Binary_All_CGCv.96.csv`
- `CD_NG_Training_Validation_Binary_Liver_specificII.txt`
- `diagnostic_file.tsv`

They are used to describe which genes are treated as cancer drivers or neutral genes in different scenarios.

Examples of usage:

- pan-cancer setting: Cancer Gene Census based labels,
- liver-specific setting: CancerMine-derived labels.

Typical columns include:

- gene name,
- target label,
- genomic location in hg38,
- source,
- strand.

The file `diagnostic_file.tsv` is used together with `Epigenetics_newgenes.py` to extract per-gene signal windows from chromosome-level files.

### 3. Numpy arrays

The folder `Numpy_arrays/` contains saved arrays used for evaluation and analysis.

See [Numpy arrays](Numpy_arrays/Readme.md) for details.

### 4. Model weights

The folder `Model_weights/` contains example OriGENE weight files in HDF5 format.

See [Model weights](Model_weights/Readme.md) for details.

### 5. Main manuscript figure inputs

[Main_figures/](Main_figures/) contains the inputs read by [Data-visualization.ipynb](../Data-visualization.ipynb).

| Subfolder under Main_figures/ | Manuscript figure | Contents |
|---|---|---|
| `manuscript_artwork/` | Figures 1 and 2 | Supplied manuscript artwork |
| `single_track_40kb/` | Figure 3a | Saved 40-kbp single-track predictions and labels |
| `multi_track_40kb/` | Figure 3b | Saved 40-kbp multi-track predictions and labels |
| `benchmark/` | Figure 4 | Benchmark comparison inputs |
| `robustness/` | Figure 5 | Gene-length and signal-intensity control results |
| `single_marker_5kb/` | Figure 6 | Saved 5-kbp single-marker predictions and labels |
| `transfer_40kb_paired/` | Figure 7 | Saved 40-kbp paired within-patient and transfer predictions |

Run the notebook from the repository root, with this directory structure intact. Figures 1 and 2 export the supplied artwork; Figures 3–7 use numerical inputs. The notebook produces eight PNG files because Figure 3 has separate panels a and b. Plotting requires a working TeX installation.

Figure 7 contains 32 `oof_predictions.tsv` files: four within-patient evaluations and 28 transfers from four lung/liver sources. Within-patient files are at `transfer_40kb_paired/<source>/oof_predictions.tsv`; transfer files are at `transfer_40kb_paired/<source>/cross_patient/<target>/oof_predictions.tsv`. The plotting notebook reads `y_true` and `y_prob`.

These saved results support figure reproduction without retraining. For training inputs and the mapping from training outputs to Figure 7 inputs, see [analysis scripts](../analysis_scripts/README.md). Requirements for supplementary analyses are documented separately in [Supplementary/README.md](../Supplementary/README.md).

## Notes on naming

In this project:

- `CD` means cancer driver,
- `NG` means neutral gene (reference negative class),
- `OG` means oncogene,
- `TSG` means tumor suppressor gene.


