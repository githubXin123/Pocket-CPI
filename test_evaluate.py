from model_7_2_2 import DTI_pred

import csv
import yaml
import numpy as np
import pandas as pd
import random
import os
from multiprocessing import Pool
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torch_geometric.data import Batch, Data, Dataset, DataLoader
from torchmetrics.classification import BinarySpecificity, AUROC, BinaryRecall, BinaryAccuracy, MatthewsCorrCoef, PrecisionRecallCurve
from sklearn.metrics import auc
from prefetch_generator import BackgroundGenerator


import warnings
warnings.filterwarnings('ignore', '.*TypedStorage is deprecated.*')
torch.multiprocessing.set_sharing_strategy('file_system')


def load_best_state_dict(model, log_dir):
    try:
        device_tmp = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        state_dict = torch.load(log_dir, map_location=device_tmp)
        model.load_my_state_dict(state_dict)
        print("Loaded pre-trained model with success.")
    except FileNotFoundError:
        print("Pre-trained weights not found. Training from scratch.")
    return model


# initialize model
config_raw = yaml.load(open("xxxxx/config_init_test.yaml", "r"), Loader=yaml.FullLoader)
print(config_raw)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = DTI_pred(**config_raw['model_fix'], **config_raw['model_opt']).to(device)



# load pre-trained model
log_dir = config_raw['pretrain_model_path']
model = load_best_state_dict(model, log_dir)

# load test data
def read_data1(data_path, set_type):
    data_new = pd.read_csv(data_path)
    if set_type == 'training':
        prot_name =  data_new.loc[data_new['Set_Type']=='training', 'uniprot_id'].tolist()
        smiles_list =  data_new.loc[data_new['Set_Type']=='training', 'smiles_neural'].tolist()
        label_list =  data_new.loc[data_new['Set_Type']=='training', 'label'].tolist()
        smiles_name =  data_new.loc[data_new['Set_Type']=='training', 'mol_name'].tolist()
    elif set_type == 'valid':
        prot_name =  data_new.loc[data_new['Set_Type']=='valid', 'uniprot_id'].tolist()
        smiles_list =  data_new.loc[data_new['Set_Type']=='valid', 'smiles_neural'].tolist()
        label_list =  data_new.loc[data_new['Set_Type']=='valid', 'label'].tolist()
        smiles_name =  data_new.loc[data_new['Set_Type']=='valid', 'mol_name'].tolist()
    else:
        prot_name =  data_new.loc[data_new['Set_Type']=='test', 'uniprot_id'].tolist()
        smiles_list =  data_new.loc[data_new['Set_Type']=='test', 'smiles_neural'].tolist()
        label_list =  data_new.loc[data_new['Set_Type']=='test', 'label'].tolist()
        smiles_name =  data_new.loc[data_new['Set_Type']=='test', 'mol_name'].tolist()
    return prot_name, smiles_list, label_list, smiles_name

def load_moldata(molname_tem):
    mol_attr = torch.load("xxxxx/data/bindingDB/mol_info/node_attr/" + molname_tem + ".pt")
    edge_index = torch.load("xxxxx/processing data/bindingDB/mol_info/edge_index/" + molname_tem + ".pt")
    edge_attr = torch.load("xxxxx/processing data/bindingDB/mol_info/edge_attr/" + molname_tem + ".pt")
    return mol_attr, edge_index, edge_attr

class dti_Dataset(Dataset):
    def __init__(self, device, data_path, class_type='training'):
        super(dti_Dataset, self).__init__()
        self.trans_device = device

        self.prot_name, self.smiles_list, self.label_list, self.smiles_name = read_data1(data_path, class_type)

        self.len_data = len(self.prot_name)

    def get(self, index):
    
        interaction_label = self.label_list[index]
        interaction_protname = self.prot_name[index]
        interaction_smi_name = self.smiles_name[index]
        tem_protinedex = filename_list.index(interaction_protname)

        x1, mol_edge_index1, mol_edge_attr1 = load_moldata(interaction_smi_name)

        graph_anchor = Data(x=x1, edge_index=mol_edge_index1, edge_attr=mol_edge_attr1, data_label = interaction_label)
        protein_feature = protfeat_list[tem_protinedex]
        protein_edge_index = protedge_list[tem_protinedex]
        protein_pocket = protpocket_list[tem_protinedex]
        protein_degree = protdegree_list[tem_protinedex]
        protein_pocket = protein_pocket * protein_degree
        protein_feature = protein_feature * protein_degree
        graph_protein = Data(x= protein_feature, edge_index=protein_edge_index, time = protein_pocket, protein_deree=protein_degree)


        interaction_list = []
        interaction_list.append(graph_anchor)
        interaction_list.append(graph_protein)
        return interaction_list

    def len(self):
        return self.len_data

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class dti_DatasetWrapper(object):
    def __init__(self, batch_size, num_workers, data_path):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_data_loaders(self, device, set_type='training', pin_para=True, prefetch_factor=4):
        train_dataset = dti_Dataset(device, data_path=self.data_path, class_type=set_type)
        print(set_type,'Number:', len(train_dataset))
        self.prefetch_factor = prefetch_factor
        train_loader = self.get_train_validation_data_loaders(train_dataset, pin_para)
        return train_loader

    def get_train_validation_data_loaders(self, train_dataset, pin_para):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))

        train_sampler = SubsetRandomSampler(indices)

        train_loader = DataLoaderX(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                num_workers=self.num_workers, drop_last=False, pin_memory=pin_para, persistent_workers=False, prefetch_factor=self.prefetch_factor)

        return train_loader


# Test
CUDA_VISIBLE_DEVICES=1

def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)

def pocket_cut(tensor_pocket):
    seq_len, _ = tensor_pocket.shape
    if seq_len > 1022:
        tensor_pocket = tensor_pocket[:1022, :]
    else:
        tensor_pocket = tensor_pocket
    return tensor_pocket


# protein_feat
def readfeat_cpu(name_len_list):
    protein_feat = np.load('/home/xxyu/DTI_predict/our_code/data/bindingDB/protein_info/node_emb/' + name_len_list[0] + '.npy',allow_pickle=True)
    protein_feat = torch.from_numpy(protein_feat).float()

    protein_feat = protein_feat.squeeze()
    if name_len_list[1] > 1022:
        num_residue = 1022
    else:
        num_residue = name_len_list[1]

    protein_feat = protein_feat[1:num_residue+1,:]

    assert protein_feat.size(0) == num_residue
    return protein_feat

# protein_edge
def readedge_cpu(filename_tem):
    protein_edge = torch.load('xxxxx/processing data/bindingDB/protein_info/contact_map/' + filename_tem + '.pt')
    return protein_edge

# residue_degree
def readdegree_cpu(filename_tem):
    protein_degree = torch.load('xxxxx/processing data/bindingDB/protein_info/prot_degree/' + filename_tem + '.pt')
    return protein_degree

# protein_pocket
def readpocket_cpu(filename_tem):
    protein_pocket = np.load('xxxxx/processing data/bindingDB/protein_info/prot_pocket/' + filename_tem + '_pocket.npy',allow_pickle=True)
    protein_pocket = torch.from_numpy(protein_pocket).float().T
    protein_pocket = pocket_cut(protein_pocket)
    return protein_pocket


def set_random_seed(seed=2024):
    print('random seed：', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


uniprot_df = pd.read_csv('xxxxx/Data/bindingDB/bindingDB_protein_info_test.csv')
filename_list = uniprot_df.loc[uniprot_df['Set_Type']=='test', 'uniprot_id'].tolist()
prot_seq_list = uniprot_df.loc[uniprot_df['Set_Type']=='test', 'BindingDB Target Chain Sequence'].tolist()

name_len_list = []
for i in range(len(filename_list)):
    name_len_list.append((filename_list[i],len(prot_seq_list[i])))


with Pool(4)as proc:
    protfeat_list = list(
        tqdm(
            proc.imap(readfeat_cpu, name_len_list,
                    ),
            total=len(name_len_list)
        ))
with Pool(4)as proc:
    protpocket_list = list(
        tqdm(
            proc.imap(readpocket_cpu, filename_list,
                    ),
            total=len(filename_list)
        ))
with Pool(4)as proc:
    protedge_list = list(
        tqdm(
            proc.imap(readedge_cpu, filename_list,
                    ),
            total=len(filename_list)))
with Pool(4)as proc:
    protdegree_list = list(
        tqdm(
            proc.imap(readdegree_cpu, filename_list,
                    ),
            total=len(filename_list)))


set_random_seed(2024)
csv_data_path = '/home/xxyu/DTI_predict/our_code/data/bindingDB/bindingDB_split_cluster.csv'
test_dataset = dti_DatasetWrapper(batch_size=64, num_workers=4, data_path=csv_data_path)
# test_dataset = dti_DatasetWrapper(batch_size=64, num_workers=6, data_path=csv_data_path)



# 预测结果
metric_auroc_func = AUROC(task="binary")
metric_spec_func =  BinarySpecificity()
metric_recall_func = BinaryRecall()
metric_acc_func = BinaryAccuracy()
metric_mcc_func =  MatthewsCorrCoef(task="binary")
metric_prauc_func =  PrecisionRecallCurve(task="binary")

criterion_eval = nn.BCELoss()

def loss_function1(pred, data_label):
    loss = criterion_eval(pred, data_label)
    return loss

def compute_metric(pred, label):
    metric_auroc = metric_auroc_func(pred, label)
    metric_recall = metric_recall_func(pred, label)
    metric_spec = metric_spec_func(pred, label)
    metric_acc = metric_acc_func(pred, label)
    metric_mcc = metric_mcc_func(pred, label)
    precision_tmp, recall_tmp, _ = metric_prauc_func(pred, label)
    metric_prauc = auc(recall_tmp, precision_tmp)
    return metric_auroc, metric_recall, metric_spec, metric_acc, metric_mcc, metric_prauc

def valid_step(model, valid_loader):
    device_tmp = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():   #with torch.no_grad()则主要是用于停止gradient计算
        model.eval()
        predictions = []
        labels = []
        label_pred = []
        valid_loss = 0.0
        num_data = 0

        
        for i, data in enumerate(valid_loader):
            for j in range(2):
                data[j] = data[j].to(device_tmp)
                # data[j] = data[j].to(self.device, non_blocking = True)

            pred = model(data[0], data[1], config_raw['weight_pocket'])

            loss = loss_function1(pred, data[0].data_label.unsqueeze(1).float())
            valid_loss += loss.item() * data[0].data_label.size(0)
            num_data += data[0].data_label.size(0)

            predictions.extend(pred.cpu().detach().flatten().numpy())
            labels.extend(data[0].data_label.cpu().flatten().numpy())
            valid_loss /= num_data
    model.train()


    predictions = torch.tensor(predictions)
    labels = torch.tensor(labels)
    auroc_score, reacall_score, spec_score, acc_score, mcc_score, prauc_score = compute_metric(predictions, labels)

    print('valid loss:', valid_loss, 'valid ROC AUC:', auroc_score, 'valid Recall:', reacall_score, 'valid Spec:', spec_score, 'valid ACC:', acc_score, 'valid MCC:', mcc_score, 'valid PRAUC:', prauc_score)    
    return valid_loss, auroc_score


dataloader_valid = test_dataset.get_data_loaders(device, set_type='test', pin_para=False, prefetch_factor= config_raw['prefetch_factor'])
# dataloader_test = test_dataset.get_data_loaders(device, set_type='test', pin_para=False)

_, roc_valid = valid_step(model, dataloader_valid)













