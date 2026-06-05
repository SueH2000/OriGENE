# Mutation feature pipeline

This folder contains the Snakemake workflow that converts VCF files into gene-level mutation feature matrices for OriGENE.

## What this workflow does

It covers:

```text
raw VCF
-> chromosome-name normalization
-> primary chromosome filtering
-> VEP annotation with dbNSFP
-> mutation feature extraction
-> mutation_features_<sample>.csv
```

## Why this workflow exists

Raw VCF records are not directly usable by the model. The mutation-aware OriGENE branch needs one numeric feature vector per gene.

This workflow creates those gene-level features in a reproducible way.

## Important files

- `workflow/Snakefile`
- `config/config.yaml`
- `config/samples.tsv`
- `scripts/mutation_feature_extractor.py`
- `envs/mutation_features.yaml`

## Important note about sample tables

This workflow has its own `config/samples.tsv`.

That file is independent from the sample tables used by:

- `epigenetic_snakemake_pipline/`
- `mutation(VCF)_snakemake_pipeline/`

In other words, each Snakemake workflow in this repository keeps its own sample sheet for its own inputs.

## Required inputs

You need:

1. VCF files, preferably compressed and indexed,
2. a CDS length reference file,
3. Ensembl VEP offline resources,
4. dbNSFP plugin data.

The bundled CDS reference is:

```text
data/reference/Final_CDS_lengths.csv
```

## Current default sample path

The example `config/samples.tsv` contains an original project-specific VCF path example for this workflow.

Before running on your own data, replace that path with the real VCF location you want this workflow to consume.

## VEP requirements

This workflow expects Ensembl VEP to run in offline mode with:

- a cache directory,
- a matching reference FASTA,
- dbNSFP data.

These settings are defined in `config/config.yaml`.

## How to run

From this folder:

```bash
conda create -n snakemake -c conda-forge -c bioconda snakemake
conda activate snakemake
snakemake -s workflow/Snakefile --configfile config/config.yaml -n -p
snakemake -s workflow/Snakefile --configfile config/config.yaml --use-conda --cores 4
```

## Main outputs

- `results/mutation_work/`: intermediate VCF processing files.
- `results/mutation_features/`: final per-gene feature tables.

Each final output file is named:

```text
mutation_features_<sample>.csv
```

## What the extractor script computes

The script `scripts/mutation_feature_extractor.py` computes 28 gene-level features, including:

- mutation density measures,
- missense and loss-of-function counts,
- silent and nonsilent ratios,
- recurrence-related features,
- PolyPhen-based impact summaries.

## Integration with OriGENE

These features are intended to be aligned by gene with the broader OriGENE modeling workflow.

Consistent gene ordering is essential.
