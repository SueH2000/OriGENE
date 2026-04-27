# Introduction

A Snakemake workflow was developed to automate the extraction of mutation-derived features from genomic variant data. The pipeline converts raw Variant Call Format (VCF) files into gene-level feature matrices that can be directly integrated with the OriGENE model.

In contrast to the epigenetic preprocessing workflow, which processes sequencing signal data, this pipeline summarizes discrete mutation events into biologically meaningful descriptors such as mutation burden, functional impact, and recurrence patterns.

Note: This workflow reproduces the mutation feature extraction used in the OriGENE-Mut model. It assumes that the input VCF files correspond to the same gene set used in the epigenetic preprocessing.

## About Snakemake

Snakemake is a workflow management system designed to ensure reproducibility and scalability of scientific pipelines.

Key properties:

Executes steps based on output targets (dependency-driven execution)
Avoids recomputation of completed steps
Automatically handles parallelization
Supports reproducible environments via Conda

If execution is interrupted, the workflow resumes from the last completed step.

## Installation

1) First install Conda (https://docs.conda.io/).

Then install Snakemake:

conda create -n snakemake -c conda-forge -c bioconda snakemake
conda activate snakemake

2) Downloading VEP Offline Resources

The pipeline requires Ensembl VEP to run in offline mode, which depends on a local cache, reference genome, and optional plugin data (dbNSFP).

All resources must match the same genome build (GRCh38) and Ensembl version (v113).

1. Install VEP (via Conda)

The workflow uses the VEP binary from the Snakemake conda environment.

If needed manually:

conda install -c bioconda ensembl-vep=113
2. Locate vep_install

After Snakemake creates the environment, locate the installer:

find .snakemake/conda -name vep_install

Typical path:

.snakemake/conda/<hash>/share/ensembl-vep-113.x/vep_install
3. Download VEP Cache and FASTA

Run:

vep_install \
  --AUTO cfp \
  --SPECIES homo_sapiens \
  --ASSEMBLY GRCh38 \
  --CACHE_VERSION 113 \
  --CACHEDIR resources/vep_cache \
  --FASTA

This downloads:

resources/vep_cache/
└── homo_sapiens/
    └── 113_GRCh38/
        ├── Homo_sapiens.GRCh38.dna.primary_assembly.fa
        ├── *.fa.fai
        └── cache files
4. Download dbNSFP (for VEP plugin)

3) Download the BGZF version (required for VEP):

mkdir -p resources/dbnsfp
cd resources/dbnsfp

wget https://download.genos.us/dbnsfp/academic/ac0ddc/BGZF_format/dbNSFP5.1a_grch38.gz
wget https://download.genos.us/dbnsfp/academic/ac0ddc/BGZF_format/dbNSFP5.1a_grch38.gz.tbi

Do not use the full ZIP version—VEP requires the BGZF-compressed file.
# Workflow overview

The mutation feature extraction pipeline consists of three stages:

Raw VCF (.vcf.gz)
→ chromosome normalization
→ primary chromosome filtering (GRCh38)
→ VEP annotation (with dbNSFP)
→ mutation feature extraction
→ mutation_features_<sample>.csv

# Directory structure
mutation_pipeline/
├── workflow/
│   └── Snakefile
├── scripts/
│   └── mutation_feature_extractor.py
├── config/
│   └── config.yaml
├── envs/
│   └── mutation_features.yaml
├── data/
│   └── reference/
│       └── Final_CDS_lengths.csv
├── resources/
│   ├── vep_cache/
│   └── dbnsfp/
└── results/
    ├── mutation_work/
    └── mutation_features/

## Input data

The pipeline requires the following inputs:

1) Raw VCF files
Format: .vcf.gz
Must be indexed with tabix
2)  CDS length reference
data/reference/Final_CDS_lengths.csv

Required columns:

Gene, CDS_length

This file defines:

the gene set
the gene order
normalization for mutation density
3)  VEP resources
VEP cache (GRCh38, version 113)
resources/vep_cache/homo_sapiens/113_GRCh38/
Reference FASTA
Homo_sapiens.GRCh38.dna.primary_assembly.fa
dbNSFP database (BGZF format)
resources/dbnsfp/dbNSFP5.1a_grch38.gz
resources/dbnsfp/dbNSFP5.1a_grch38.gz.tbi

### All parameters are defined in:

config/config.yaml

This includes:

input VCF paths
VEP configuration (cache, FASTA, dbNSFP)
output directories
sample definitions
Running the workflow

### Execute from the project root:

snakemake -s mutation_pipeline/workflow/Snakefile \
          --configfile mutation_pipeline/config/config.yaml \
          --cores 4 \
          --use-conda
## Workflow steps
1) Chromosome Normalization

Chromosome prefixes ("chr") are removed when necessary to ensure compatibility with VEP reference files.

Example:

zcat input.vcf.gz | sed 's/^chr//' | bgzip > sample.nochr.vcf.gz
2) Primary Chromosome Filtering

Keeps only chromosomes present in the GRCh38 primary assembly FASTA.

This removes:

alternative contigs
decoy sequences
random scaffolds

This step ensures compatibility with VEP and prevents annotation warnings.

3) Variant annotation using (VEP)

Variants are annotated using Ensembl VEP in offline mode with cache support.

The annotation includes:

gene symbol assignment (--symbol)
canonical transcript selection (--canonical --pick)
functional prediction using dbNSFP

The dbNSFP plugin extracts the following scores:

Polyphen2_HDIV_score
Polyphen2_HVAR_score
REVEL_score
SIFT_score
MutationTaster_score
MetaLR_score
MetaSVM_score
3) Mutation feature extraction

Annotated VCF files are processed using:

scripts/mutation_feature_extractor.py

This script computes 28 gene-level mutation features, including:

mutation densities (per kilobase)
mutation type counts and ratios
functional impact ratios (HiFI / LoFI)
missense entropy
recurrent mutation fraction
Recurrent missense definition

Recurrent missense fraction is defined as:

the proportion of missense mutations occurring at protein positions with count > 1.

## Output

results/mutation_features/
├── mutation_features_<sample>.csv

Each file contains a gene-level feature matrix aligned by gene symbol.

# Reproducibility notes
VEP runs in offline mode with fixed cache (v113)
dbNSFP version is fixed (5.1a)
CDS length is used to normalize mutation densities
ratio-based features use a pseudocount of 1
gene names are normalized to uppercase for consistency


# Integration with OriGENE

The mutation feature matrix is designed to integrate with OriGENE as follows:

x_epi[i] ↔ x_mut[i] ↔ y[i] ↔ g[i]

Ensuring consistent gene ordering is essential for correct model training.

This pipeline provides a reproducible and modular approach for extracting mutation-derived features from genomic data. It is designed to complement the OriGENE epigenetic pipeline and enable integrated modeling of genomic and epigenomic signals.
