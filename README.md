# Pocket-CPI
Pocket-CPI: an explainable deep learning approach for prediction of compound-protein interactions by integration of protein structures and binding pocket information


## Requirements
  * python = 3.8
  * pytorch = 2.0
  * torch geometric
  * rdkit
  * numpy
  * pandas
  * prefetch-generator
  * pyyaml
  * tensorboard
  * ESM-2
  * PocketMiner

## 🛠️ Data Preparation & Processing
- Download three raw datasets (BIOSNAP, Metz, and BindingDB) from Dataset Source and place them in the Data directory
- Obtain protein PDB files from PDB Source and store them in the pdb_file directory

### Residue-level features
python extract_protein_feature.py

### Contact map features
python extract_protein_contactMap.py

### Residue degree features
python extract_protein_degree.py

### Pocket features (requires PocketMiner)
cd PocketMiner/src/
python extract_protein_pocket.py

Note: Protein pocket embeddings are calculated using the methodology from PocketMiner Reference

## Model Training
python model_test.py

## Model Test
python model_test.py




