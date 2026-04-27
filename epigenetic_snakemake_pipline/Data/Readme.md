# Data

### *1. Dataset information and Performance Metrics*

Data about Gene Expression Omnibus accession names for the used files and their properties: number of reads, names, markers, quality of the mapping...
Tables with performance metrics for OriGENE in differen scenarios can also be found here.

### *2. Gene target labels and information*

The files CD_Binary_All_CGCv.96.csv and CD_NG_Training_Validation_Binary_Liver_specificII.txt contain information about the gene sets used in different contexts:
- Pan-cancer study: CGC v96 genes.
- Liver-specific study: Cancermine genes 

    
These files, which were used in order to retrieve the original input sequences, contain the following information in columns:
- **Gene name** : name as it can be found in the Genome Browser: <https://genome.ucsc.edu/>. Original paper: Kent WJ, Sugnet CW, Furey TS, Roskin KM, Pringle TH, Zahler AM, Haussler D. *The human genome browser at UCSC.* Genome Res. 2002 Jun;12(6):996-1006. 
    
- **Target category or label**: 
  -  CDs (TSG/OGs) were obtained from The Cancer Gene Census, a highly curated database ( <https://cancer.sanger.ac.uk/census>)
  -  Housekeeping, Neutral Genes (NGs) were obtained from DORGE, Lyu et al., Sci. Adv. 2020; 6 : eaba6784     11 November 2020. They were a further curated set of the genes found in T. Davoli et al., *Cumulative haploinsufficiency and triplosensitivity drive aneuploidy patterns and shape the cancer genome.* Cell 155, 948–962 (2013). <https://doi.org/10.1016/j.cell.2013.10.011>.
  - For the liver-specific scenario, the gene names were obtained from  J. Lever et al., *CancerMine: a literature-mined resource for drivers, oncogenes and tumor suppressors in cancer.* Nature Methods volume 16, pages 505–507 (2019). <http://bionlp.bcgsc.ca/cancermine>

    
- **Location in the genome**: Location in the Human reference genome assembly hg38 given as chr{Number}:{Leftmost basepair}-{Rightmost basepair}

- **Source**: Where was the information found
- **Strand**: 'pos' or 'neg'

### *3. Numpy arrays*

Numpy arrays containing different types of data.

### *4. Model weights*

.hdf5 files containing weights for OriGENE in different scenarios.
