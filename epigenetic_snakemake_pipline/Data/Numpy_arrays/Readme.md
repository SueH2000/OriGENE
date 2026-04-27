
# <center> Numpy arrays <center>

- ypred files: arrays with the predicted probabilities of each gene to be a CD
- ytrue files: arrays with gene class. 1 for CDs, 0 for NGs
- gene files: files with the list of genes in the order in which they appear in ypred and ytrue files. Genes have been reordered for the stratified k-fold partitioning  of the data. Validation genes (stratified) have been gathered for each fold and appended to the final array in order.
