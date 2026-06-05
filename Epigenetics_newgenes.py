#!/usr/bin/env python
# coding: utf-8



import os
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import tensorflow as tf
import time


def create_txt(Diagnostic_file,PATH,outPATH, shift = 2000):
    
    """
    This function is aimed to read the Diagnostic_file (Table file with the relevant information of all the genes, e.g. 
    TSG_OG_NG_Test.txt) and extract the ChIP sequences as txt files in the folder outPATH:

    
                                      __________ Inputs_____________
    
    Diagnostic_file: txt/csv file with columns # Gene name; Label, Location; Source of information; Strand
    PATH: Path where we have the files for each chromosome.
    outPATH: Path where we'll create the txt files with the same columns but ranging accordingly to each gene. 
             IMPORTANT: The folder should end with the name of the epigenetic marker that we are using,
             e.g H3K4me1 ...
    shift:  Number of basepairs before the beginning (and after the end) of the gene that we want: critical to find features.
            Normally set to 1000 or 2000.
    
                                      _________ Outputs ____________
                                      
    The function does not return anything, just creates the files for each gene.
    
    """ 
    Cell_line = PATH.split('/')[-4]
    sample = PATH.split('/')[-2].split('_')[0] #Name of the sample
    epi_marker = outPATH.split('/')[-2].split('_')[-1] # Name of the epigenetic marker
    chr_directory = [f.name for f in os.scandir(PATH)]  #list of chromosome folders
    Target_list = []
        
    with open(Diagnostic_file,'r') as diagnosticFile:   
        
    # Creating targets and reading Diagnostic_file
        for line in diagnosticFile.readlines():
            if line[0] != "#":
                #splitline = line.split('\t |\n')
                splitline = line.split()
                Gene_name = splitline[0]
                Diagnostic = splitline[1] 
                Strand = splitline[4]
                print(splitline)
                 
    # Retrieving information from the corresponding chromosome file
                chrsplit = splitline[2].split(':')
                chr_num = chrsplit[0]              # Chromosome number
                basepairs = chrsplit[1].split('-') # Range in basepairs of the chromosome, separated first and last
                first_basepair = basepairs[0].replace(',','')
                last_basepair = basepairs[1].replace(',','')
                first_basepair = str(int(first_basepair) - shift)
                last_basepair = str(int(last_basepair) + shift) # We can also crop or not the last part of the gene
                
                for filename in chr_directory: # For each document we will retreive the data for the position x and the signal value y.
                    if filename.split('_')[-1][:-4] == chr_num: # without '.txt', chromosome where we find the desired gene
                        with open(PATH+filename,'r') as inputFile, open (outPATH+'{}_{}_Shifted_{}_{}_{}_{}_{}.txt'.format(Cell_line,sample,shift,epi_marker,Gene_name,Strand,Diagnostic),'w') as outputFile:
                            outputFile.write('# Cell line {}, Gene {}, Diagnostic {} \n'.format(Cell_line,Gene_name,Diagnostic))
                            #In case the marker had no reads in this gene, we add zeros just before and after the gene.
                            outputFile.write('{}\t{}\t{}\t{}\n'.format(chr_num,int(first_basepair)-2, int(first_basepair)-1, 0.))
                            for line in inputFile.readlines():
                                if line[0] == 'c':
                                    splitline = line.split()
                                    if float(splitline[1]) >= float(first_basepair) and  float(splitline[1]) <= float(last_basepair):
                                        outputFile.write(line)
                            outputFile.write('{}\t{}\t{}\t{}\n'.format(chr_num,int(last_basepair)+1, int(last_basepair)+2, 0.))
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
################################################################################################################################           

############################################ Usage example #####################################################################                                        


# Path where we have the chromosome txt files for a certain cell line and epigenetic marker:
PATH = '/scratch/lema/marc/oncogenes/GSE31755/chromosomes_rmdup/GSM788071_chr/'
# Path where we will create txt files gene by gene of a certain epigenetic marker:
outPATH = '/scratch/lema/marc/oncogenes/GSE31755/Combined_txt_files_NT2D1_rmdup/txt_files_NT2D1/txt_files_H3K27me3/'
# Table sheet files containing the diagnostic or target category of each gene
Diagnostic_file = '/home/william/marc/marc_directory/OriGENE_paper/Spreadsheets/CD_Binary_All_CGCv.96.csv'

Target = create_txt(Diagnostic_file,PATH,outPATH)
