# Variant-calling pipeline

This folder contains the Snakemake workflow that converts raw sequencing input into compressed, indexed VCF files.

## What this workflow does

It covers:

```text
SRA or FASTQ
-> FASTQ preparation
-> Bowtie2 alignment
-> sorted BAM
-> duplicate-filtered BAM
-> BAM indexing
-> bcftools variant calling
-> compressed VCF plus tabix index
```

It does not compute mutation features directly. Its job is to produce VCF files for the downstream mutation feature pipeline.

## Why this workflow exists

The mutation-aware OriGENE branch needs variant calls before it can build gene-level mutation features.

## Important files

- `workflow/Snakefile`
- `config/config.yaml`
- `config/samples.tsv`
- `envs/`

## Input formats

Samples are defined in `config/samples.tsv`.

Required columns:

| Column | Meaning |
|---|---|
| `sample` | Sample name used in output filenames |
| `input_type` | One of `sra`, `fastq_single`, or `fastq_paired` |
| `read1` | SRA accession or FASTQ path |
| `read2` | Empty for single-end input, or paired-end R2 FASTQ path |

## Configuration

Edit `config/config.yaml` before running.

Important settings:

- `samples_file`
- `genome`
- `fastq_dir`
- `bam_dir`
- `dedup_bam_dir`
- `vcf_dir`
- `bowtie_threads`
- `samtools_threads`
- `bcftools_threads`

The current default VCF output directory is:

```text
results/04_called_variants
```

## Reference genome

Current default:

```text
resources/genome/hg38.fa
```

## How to run

From this folder:

```bash
conda create -n snakemake -c conda-forge -c bioconda snakemake
conda activate snakemake
snakemake -s workflow/Snakefile --configfile config/config.yaml -n -p
snakemake -s workflow/Snakefile --configfile config/config.yaml --use-conda -j 6 -p
```

## Main outputs

- `results/01_fastq_files/`
- `results/02_bam_sorted_files/`
- `results/03_bam_deduplicated_files/`
- `results/04_called_variants/`

Each sample should produce:

- `<sample>.vcf.gz`
- `<sample>.vcf.gz.tbi`

## Downstream use

The expected next step is:

```text
VCF -> VEP annotation -> mutation feature extraction
```
