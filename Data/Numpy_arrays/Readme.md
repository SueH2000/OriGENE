# Numpy arrays

This folder contains saved arrays used for analysis and evaluation.

## File types

- `*_ypred.npy`: predicted probability that each gene is a cancer driver.
- `*_ytrue.npy`: true label for each gene, where `1` means cancer driver and `0` means normal or neutral gene.
- `*_genes.npy`: ordered list of gene names aligned with `ypred` and `ytrue`.

## Why the gene order matters

The prediction arrays and label arrays are meaningful only if they are read in the same order as the gene list.

In several experiments, genes were reordered during stratified cross-validation. The saved `*_genes.npy` file preserves that order.
