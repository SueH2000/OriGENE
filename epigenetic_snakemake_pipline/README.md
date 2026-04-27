# Epigenomic Signal Generation for OriGENE

This directory contains the Snakemake workflow used to convert raw ChIP-seq reads into genome-wide epigenomic signal files for downstream OriGENE preprocessing.

The workflow preserves the original OriGENE signal-generation logic: raw sequencing reads are aligned to the reference genome, converted into duplicate-filtered BAM files, transformed into bedGraph coverage, and sorted into final `*.bedgraph.sorted` files.

The original project description stated that the ChIP-seq raw reads were re-expressed as continuous enrichment levels in `bedgraph.sorted` files. This workflow formalizes that step and adds support for multiple input formats.

## Scope

This workflow performs **signal generation only**.

It covers:

```text
SRA / FASTQ
→ FASTQ
→ Bowtie2 alignment
→ sorted BAM
→ duplicate-removed BAM
→ bedGraph
→ bedGraph.sorted
```

It does **not** perform gene-centered extraction, padding/cropping, NumPy array construction, model training, or visualization. Those steps belong to the downstream OriGENE model-preparation code.

## Directory structure

```text
epigenomic_signal_generation/
├── workflow/
│   └── Snakefile
├── config/
│   ├── config.yaml
│   └── samples.tsv
├── envs/
│   ├── bowtie2.yaml
│   ├── htslib.yaml
│   ├── sra_tools.yaml
│   ├── bedtools.yaml
│   └── ucsc.yaml
├── data/
├── resources/
│   └── genome/
│       └── hg38.fa
└── results/
    
```

## Input formats

Samples are defined in `config/samples.tsv`.

Required columns:

| Column | Description |
|---|---|
| `sample` | Unique sample name used in output filenames |
| `input_type` | One of `sra`, `fastq_single`, or `fastq_paired` |
| `read1` | SRA accession/local SRA file, single-end FASTQ, or paired-end R1 FASTQ |
| `read2` | Empty for `sra` and `fastq_single`; paired-end R2 FASTQ for `fastq_paired` |

Supported `input_type` values:

| input_type | Meaning |
|---|---|
| `sra` | Local SRA file or SRA accession readable by `fastq-dump` |
| `fastq_single` | Single-end FASTQ or FASTQ.GZ file |
| `fastq_paired` | Paired-end FASTQ or FASTQ.GZ files |

Example:

```text
sample	input_type	read1	read2
SRR1947881.1	sra	data/raw/SRR1947881.1	
example_single	fastq_single	data/raw/example_single.fastq.gz	
example_paired	fastq_paired	data/raw/example_R1.fastq.gz	data/raw/example_R2.fastq.gz
```

SRA input is treated as single-end to preserve the historical workflow. For paired-end SRA datasets, dump the reads externally into paired FASTQ files and use `input_type=fastq_paired`.

## Configuration

Edit `config/config.yaml` before running.

Important fields:

| Field | Description |
|---|---|
| `samples_file` | Path to the sample table |
| `genome` | Reference genome FASTA used by Bowtie2 |
| `fastq_dir` | FASTQ output directory |
| `sorted_bam_dir` | Sorted BAM output directory |
| `dedup_bam_dir` | Duplicate-filtered BAM output directory |
| `bedgraph_dir` | bedGraph output directory |
| `sorted_bedgraph_dir` | Final sorted bedGraph output directory |
| `bowtie_threads` | Number of threads for Bowtie2 alignment |
| `samtools_threads` | Number of threads for BAM indexing |
| `remove_duplicates` | Whether to run historical `samtools rmdup -s` |

## Reference genome

Place the reference genome FASTA under `data/reference/`, for example:

```text
data/reference/hg38.fa
```

The `genome` path in `config/config.yaml` must point to this file.

The Bowtie2 index is generated automatically next to the FASTA by the `bowtie_index` rule if the index files are missing.

## Running the workflow

From the `epigenomic_signal_generation/` directory:

```bash
conda create -n snakemake -c conda-forge -c bioconda snakemake
conda activate snakemake

snakemake -s workflow/Snakefile -p --use-conda -j 6
```

For a dry run:

```bash
snakemake -s workflow/Snakefile -n -p
```

## Final output

The final output is:

```text
results/epigenetic_work/bedgraph_sorted/{sample}.bedgraph.sorted
```

Each sorted bedGraph file has four columns:

| Column | Meaning |
|---|---|
| 1 | Chromosome |
| 2 | Start genomic coordinate |
| 3 | End genomic coordinate |
| 4 | Raw read coverage signal |

Example:

```text
chr1    10000   10007   1
chr1    10007   10029   2
chr1    10029   10030   3
chr1    10030   10031   4
```

Signal values are raw coverage values. Any further normalization, gene-level alignment, missing-position filling, padding/cropping, and tensor construction should be documented in the downstream OriGENE preprocessing code.

## Notes on duplicate removal

The workflow uses `samtools rmdup -s` by default to preserve the historical OriGENE preprocessing logic.

This is retained for replication. For a new production ChIP-seq workflow, `samtools fixmate` followed by `samtools markdup` would usually be preferable, especially for paired-end reads. Changing this behavior may change the resulting signal files, so it should not be done silently when the goal is replication.

## Data source and citation

The original OriGENE epigenomic analysis used public ChIP-seq datasets from GEO, including GSE67471 from Chen et al. Users should cite the original data generators when reusing these datasets.
