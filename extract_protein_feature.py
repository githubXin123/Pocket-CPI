import esm
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
import torch
import os
from Bio import SeqIO
from typing import List, Tuple
import string
import torch.nn.functional
import numpy as np
import random
from torch import nn


# load data
import pandas as pd 
def load_data(path):
    data_frame = pd.read_csv(path)
    prot_name_list = data_frame['uniprot_id'].tolist()
    prot_seq_list = data_frame['seq'].tolist()

    return prot_name_list, prot_seq_list

# using ESM to extract features
CUDA_VISIBLE_DEVICES=1

esm1b, esm1b_alphabet = esm.pretrained.esm2_t33_650M_UR50D()


esm1b = esm1b.eval().cuda()
esm1b_batch_converter = esm1b_alphabet.get_batch_converter()
list1=[]

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result

def set_diagonal_to_one(tensor_1):
    tensor_1 = torch.squeeze(tensor_1)
    # diagonal_tensor = torch.eye(tensor_1.size(0))
    # result = tensor_1 * (1 - diagonal_tensor) + diagonal_tensor
    result = tensor_1.fill_diagonal_(1)
    result = result.unsqueeze(0)
    return result


def protein_embedding(path):
    prot_name, prot_seq = load_data(path)

    for i,j in enumerate(prot_seq):
        # print(j)
        esm1b_data = [
        (str(prot_name[i]), str(j)),
        ]

        esm1b_batch_labels, esm1b_batch_strs, esm1b_batch_tokens = esm1b_batch_converter(esm1b_data)
        b =esm1b_batch_tokens.shape[1:3][0]
        if b > 1024:
            tokens = torch.cat([esm1b_batch_tokens[:,0:1023], esm1b_batch_tokens[:,-1:]],dim=1)
        elif b < 1024:
            tokens_pad = torch.nn.functional.pad(esm1b_batch_tokens[:,:b-1],pad=(0,1024-b,0,0), mode='constant', value=1)

            tokens = torch.cat([tokens_pad, esm1b_batch_tokens[:,-1:]],dim=1)
        else:
            tokens = esm1b_batch_tokens

        tokens = torch.tensor(tokens.numpy())
        
        with torch.no_grad():
            tokens=tokens.cuda()
            results = esm1b(tokens, repr_layers=[33], return_contacts=True)         
            token_representations = results["representations"][33].cpu()   
            token_representations.squeeze()  

        torch.cuda.empty_cache()
        
        np.save('/xxxxx/processing data/bindingDB/prot_info/node_emb/'+str(prot_name[i])+'.npy',token_representations.numpy())
        # np.save('/HOME/scz0500/run/DTI_predict/data/case_sutdy_1/prot_info/contact_map/'+str(prot_name[i])+'.npy',esm1b_contacts)

        
    torch.cuda.empty_cache()
    
    
if __name__ == "__main__":   
    protein_embedding('xxxxx/Data/bindingDB/bindingDB_protein_info.csv')


















