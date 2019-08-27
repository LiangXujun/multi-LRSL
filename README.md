# multi-LRSL
This folder contains the code and data for the paper 'Learning important features from multi-view data to predict drug side effects'.

'mlrsl_all_feature.py' is the python code for the proposed algorithm. The code was tested under the environment of python 3.7.3 + numpy 1.16.3 + pandas 0.24.2 + scipy 1.2.1 + scikit-learn 0.20.0. 


501 drugs with chemical structures, target domains, target gene ontology terms (biological process terms only), gene expression profiles and known side effects were used as golden standard dataset. All drugs are indicated by their DrugBank IDs. 

'drug_fp_mat.txt' contains the PubChem fingerprints of the drugs. The rows of the data table are the drugs and the columns are pubchem fingerprints. The description of the substructure fingerprints could be found at ftp://ftp.ncbi.nlm.nih.gov/pubchem/specifications/pubchem_fingerprints.txt

'drug_domain_mat.txt' contains the protein domain information of the drug targets related to human. The human target proteins of drugs were extracted from DrugBank(https://www.drugbank.ca/). The rows are drugs and the columns are the protein domains. The raw data were obtained from InterPro database(http://www.ebi.ac.uk/interpro/download.html). The description of the protein domains could also be found there.

'drug_gobp_mat.txt' contains the gene ontology terms(biological process terms only) of the drug targets related to human. The rows are drugs and the columns are the GO BP terms. The raw data were obtained from ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/HUMAN_9606_idmapping_selected.tab.gz.
The description of the GO BP terms could be found at http://geneontology.org/

'drug_gene_expr_mat.txt' contains the gene expression profiles related to the drugs. The rows are drugs and the columns are genes. The raw data were from LINCS L1000 projects(https://clue.io) and two related datasets were downloaded from GEO(GEO accession:GSE70138, GSE92742).The detailed description of these datasets could be found at https://clue.io/GEO-guide.

'drug_pt_mat' contains the side effects of the drugs. The rows are drugs and the columns are side effect cuis. The raw data were obtained from http://sideeffects.embl.de/download/

The other four text files 'drug_fp_mat_case_study.txt', 'drug_domain_mat_case_study.txt', 'drug_gobp_mat_case_study.txt', 'drug_gene_expr_mat_case_study.txt' contain feature matrices of 320 drugs without side effect records in SIDER database. These drugs were used in the case study of the paper.
