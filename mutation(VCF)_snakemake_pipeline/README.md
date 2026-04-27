# Variant Calling Pipeline

### Introduction

A Snakemake workflow was prepared to automate the conversion of sequencing read files into variant calls in VCF format. This workflow is the upstream step for the mutation feature extraction pipeline.

The workflow performs read conversion, genome alignment, BAM processing, duplicate removal, and variant calling. The final output is a compressed and indexed VCF file for each sample.

The pipeline supports multiple input types and produces compressed, indexed VCF files suitable for downstream annotation and mutation feature extraction.

**Note:** This workflow is intended to generate VCF files for downstream VEP annotation and mutation feature extraction. It does not compute mutation features directly.

---

## Workflow

The pipeline performs the following steps:

Input (SRA / FASTQ)
→ FASTQ preparation
→ Read alignment (Bowtie2)
→ BAM sorting
→ Duplicate removal
→ BAM indexing
→ Variant calling (bcftools)
→ Compressed VCF + index

---

## Input Data

### Reference genome

resources/genome/hg38.fa

Download by
!wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
---

### Sample table

config/samples.tsv

Format:

sample    input_type    read1    read2

Supported input types:
- sra
- fastq_single
- fastq_paired

Example:

SRR1947881    fastq_single    data/fastq/SRR1947881.fastq.gz
SRR12491691   fastq_paired    data/fastq/SRR12491691_1.fastq.gz    data/fastq/SRR12491691_2.fastq.gz

---

## Configuration

config/config.yaml

Required:

genome: resources/genome/hg38.fa

Optional:

bowtie_threads: 6  
samtools_threads: 6  
bcftools_threads: 4  

---

## Installation

conda create -n snakemake -c conda-forge -c bioconda snakemake  
conda activate snakemake  

Dependencies are installed automatically with:

--use-conda

---

## Running the Pipeline

Dry run:

snakemake -s workflow/Snakefile --configfile config/config.yaml -n -p

Run:

snakemake -s workflow/Snakefile \
  --configfile config/config.yaml \
  --use-conda \
  --conda-frontend conda \
  -j6 -p

---

## Output

results/05_called_variants/

Each sample produces:

<sample>.vcf.gz  
<sample>.vcf.gz.tbi  

---

## Directory Structure

variant_calling_pipeline/
├── workflow/
│   └── Snakefile
├── config/
│   ├── config.yaml
│   └── samples.tsv
├── envs/
├── resources/
│   └── genome/
│       └── hg38.fa
├── data/
│   └── fastq/
└── results/
---

## Downstream Use

VCF → VEP annotation → mutation feature extraction

Example:

SRR1947881    results/04_called_variants/SRR1947881.vcf.gz

---

## Notes

- Duplicate removal uses samtools rmdup
- VCF files are bgzip-compressed and tabix-indexed
- Designed for reproducible execution via Snakemake
