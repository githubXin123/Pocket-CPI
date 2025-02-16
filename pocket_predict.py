import tensorflow as tf
from models import MQAModel
import numpy as np
from glob import glob
import mdtraj as md
import os
import pandas as pd

from validate_performance_on_xtals import process_strucs, predict_on_xtals

def make_predictions(pdb_paths, model, nn_path, debug=False, output_basename=None):
    '''
        pdb_paths : list of pdb paths
        model : MQAModel corresponding to network in nn_path
        nn_path : path to checkpoint files
    '''
    strucs = [md.load(s) for s in pdb_paths]
    X, S, mask = process_strucs(strucs)
    if debug:
        np.save(f'{output_basename}_X.npy', X)
        np.save(f'{output_basename}_S.npy', S)
        np.save(f'{output_basename}_mask.npy', mask)
    predictions = predict_on_xtals(model, nn_path, X, S, mask)
    return predictions

def extract_between_symbols(text, symbol1, symbol2):
    start_index = text.find(symbol1)
    if start_index == -1:
        return None
    start_index += len(symbol1)
    end_index = text.find(symbol2, start_index)
    if end_index == -1:
        return None
    return text[start_index:end_index]


# main method
if __name__ == '__main__':
    debug = False

    # Load MQA Model used for selected NN network
    nn_path = "xxxxx/pocketminer/models/pocketminer"
    DROPOUT_RATE = 0.1
    NUM_LAYERS = 4
    HIDDEN_DIM = 100
    model = MQAModel(node_features=(8, 50), edge_features=(1, 32),
                     hidden_dim=(16, HIDDEN_DIM),
                     num_layers=NUM_LAYERS, dropout=DROPOUT_RATE)
    
    
    # if debug:
    #     output_basename = f'{output_folder}/{output_name}'
    #     predictions = make_predictions(strucs, model, nn_path, debug=True, output_basename=output_basename)
    # else:

    strucs_list_path = 'xxxxx/pdb_file'
    strucs_list = os.listdir(strucs_list_path)
    output_folder = 'xxxxx/processing data/bindingDB/protein_info/prot_pocket'

    bindingDB_df = pd.read_csv('xxxxx/Data/bindingDB/bindingDB_protein_info.csv/bindingDB_protein_info.csv')
    bindingDB_prot = bindingDB_df['UniProt_ID'].tolist()
    bindingDB_prot = [i + '.pdb' for i in bindingDB_prot]

    metz_uniq_prot = []
    for i in bindingDB_prot:
        if i not in strucs_list:
            metz_uniq_prot.append(i)


    for i,j in enumerate(metz_uniq_prot):
        singel_prot_path = os.path.join(strucs_list_path, j)
        predictions = make_predictions([singel_prot_path], model, nn_path)

        newfile_name = os.path.splitext(j)[0]
        # output filename can be modified here
        np.save(f'{output_folder}/{newfile_name}_pocket.npy', predictions)
        # np.savetxt(os.path.join(output_folder,f'{j}-predictions.txt'), predictions, fmt='%.4g', delimiter='\n')


