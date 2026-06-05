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

Two important files in this folder define gene sets and labels used in the study:

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

See `Data/Numpy_arrays/Readme.md` for details.

### 4. Model weights

The folder `Model_weights/` contains example OriGENE weight files in HDF5 format.

See `Data/Model_weights/Readme.md` for details.

## Notes on naming

In this project:

- `CD` means cancer driver,
- `NG` means normal or neutral gene,
- `OG` means oncogene,
- `TSG` means tumor suppressor gene.
