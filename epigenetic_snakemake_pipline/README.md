# Epigenetic signal pipeline

This folder contains the Snakemake workflow that converts raw ChIP-seq sequencing input into genome-wide signal tracks for OriGENE.

## What this workflow does

The workflow prepares the epigenetic input used before model training.

It covers:

```text
SRA or FASTQ
-> FASTQ preparation
-> Bowtie2 alignment
-> sorted BAM
-> duplicate-filtered BAM
-> bedGraph
-> sorted bedGraph
```

It does not perform gene-centered extraction, NumPy array creation, model training, or visualization.

## Why this workflow exists

OriGENE does not train directly on raw sequencing reads. It needs processed genome-wide signal tracks that summarize histone mark enrichment along the genome.

This workflow creates those signal tracks in a reproducible way.

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

- `samples_file`: sample table path,
- `genome`: reference FASTA path,
- `fastq_dir`: FASTQ output directory,
- `sorted_bam_dir`: sorted BAM output directory,
- `dedup_bam_dir`: duplicate-filtered BAM output directory,
- `bedgraph_dir`: bedGraph output directory,
- `sorted_bedgraph_dir`: final sorted bedGraph output directory.

## Reference genome

The workflow expects an hg38 FASTA path in `config/config.yaml`.

Current default:

```text
resources/genome/hg38.fa
```

The Bowtie2 index is generated automatically if it is missing.

## How to run

From this folder:

```bash
conda create -n snakemake -c conda-forge -c bioconda snakemake
conda activate snakemake

snakemake -s workflow/Snakefile \
  --use-conda \
  --conda-frontend conda \
  --cores 1 \
  --conda-create-envs-only

snakemake -s workflow/Snakefile \
  --use-conda \
  --conda-frontend conda \
  -n -p -j 6

snakemake -s workflow/Snakefile \
  --use-conda \
  --conda-frontend conda \
  -p -j 6
```

The LEMA workflow uses Conda rather than Mamba. Keep
`--conda-frontend conda` in every Snakemake command that enables
`--use-conda`.

## Main outputs

- `results/01_fastq_files/`
- `results/02_bam_sorted_files/`
- `results/03_bam_deduplicated_files/`
- `results/04_bedgraph_files/`
- `results/05_bedgraph_sorted_files/`

The final files are:

```text
results/05_bedgraph_sorted_files/{sample}.bedgraph.sorted
```

Each output line contains:

1. chromosome,
2. start coordinate,
3. end coordinate,
4. coverage value.

## Notes

This workflow keeps `samtools rmdup -s` as the default duplicate-removal behavior to stay close to the historical OriGENE preprocessing logic.
