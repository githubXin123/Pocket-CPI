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
- Download three raw datasets (BIOSNAP, Metz, and BindingDB) from [here](https://drive.google.com/drive/folders/1THATwG_cxQvzn4-XWf19tGT9ugwBFYVP?usp=drive_link) and place them in the `Data` directory
- Obtain protein PDB files from [here](https://drive.google.com/file/d/1jK1wDX_PQ_ha4pOgB0CWCpKSPj8Vb8to/view?usp=drive_link), unzip and store them in the `pdb_file` directory

### Residue-level features
`python extract_protein_feature.py`

### Contact map features
`python extract_protein_contactMap.py`

### Residue degree features
`python extract_protein_degree.py`

### Pocket features (requires PocketMiner)
cd PocketMiner/src/
`python extract_protein_pocket.py`

Note: Protein pocket embeddings are calculated using the methodology from [PocketMiner](https://github.com/Mickdub/gvp/tree/pocket_pred) code.

## 🛠️ Model Training
`python model_training.py`

## 🛠️ Model Test
`python model_test.py`




