import numpy as np
from datetime import datetime

import os

import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.classification import BinarySpecificity, AUROC, BinaryRecall, BinaryAccuracy, MatthewsCorrCoef, PrecisionRecallCurve
from sklearn.metrics import auc
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.tensorboard import SummaryWriter


from model import DTI_pred
from multiprocessing import Pool
from tqdm import tqdm

from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler


from torch_geometric.data import Batch, Data, Dataset, DataLoader
from torch_geometric.utils import to_dense_batch
from prefetch_generator import BackgroundGenerator


import yaml
import pandas as pd
import random

import warnings
warnings.filterwarnings('ignore', '.*TypedStorage is deprecated.*')
torch.multiprocessing.set_sharing_strategy('file_system')
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# torch.cuda.device_count()



# Load Dataset
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
    mol_attr = torch.load("xxxxx/processing data/bindingDB/mol_info/node_attr/" + molname_tem + ".pt")
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

        x1, mol_edge_index1, mol_edge_attr1 = load_moldata(interaction_smi_name)        # mol_index的结果是一个索引
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
        num_train = len(train_dataset)
        indices = list(range(num_train))

        train_sampler = SubsetRandomSampler(indices)
        train_loader = DataLoaderX(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                num_workers=self.num_workers, drop_last=False, pin_memory=pin_para, persistent_workers=False, prefetch_factor=self.prefetch_factor)

        return train_loader


# Training
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


# Pre-loadding data
## proetin_feature
def readfeat_cpu(name_len_list):
    protein_feat = np.load('xxxxx/processing data/bindingDB/protein_info/node_emb/' + name_len_list[0] + '.npy',allow_pickle=True)
    protein_feat = torch.from_numpy(protein_feat).float()

    protein_feat = protein_feat.squeeze()
    if name_len_list[1] > 1022:
        num_residue = 1022
    else:
        num_residue = name_len_list[1]

    protein_feat = protein_feat[1:num_residue+1,:]

    assert protein_feat.size(0) == num_residue
    return protein_feat

## protein_edge
def readedge_cpu(filename_tem):
    protein_edge = torch.load('xxxxx/processing data/bindingDB/protein_info/contact_map/' + filename_tem + '.pt')
    return protein_edge

## residues_degree
def readdegree_cpu(filename_tem):
    protein_degree = torch.load('xxxxx/processing data/bindingDB/protein_info/prot_degree/' + filename_tem + '.pt')
    return protein_degree

## protein_pocket
def readpocket_cpu(filename_tem):
    protein_pocket = np.load('xxxxx/processing data/bindingDB/protein_info/prot_pocket/' + filename_tem + '_pocket.npy',allow_pickle=True)
    protein_pocket = torch.from_numpy(protein_pocket).float().T
    protein_pocket = pocket_cut(protein_pocket)
    return protein_pocket


class Model_train(object):
    def __init__(self, config_raw):
        
        self.device = 'cuda:1'
        torch.cuda.set_device(self.device)

        dir_name = datetime.now().strftime('%b%d_%H-%M-%S')
        self.log_dir = os.path.join('ckpt', dir_name)

        # path of record file
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.model = DTI_pred(**config_raw['model_fix'], **config_raw['model_opt']).to(self.device)

        self.criterion = nn.BCELoss()

        self.metric_auroc = AUROC(task="binary")
        self.metric_spec =  BinarySpecificity()
        self.metric_recall = BinaryRecall()
        self.metric_acc = BinaryAccuracy()
        self.metric_mcc =  MatthewsCorrCoef(task="binary")
        self.metric_prauc =  PrecisionRecallCurve(task="binary")


    def compute_metric(self, pred, label):
        metric_auroc = self.metric_auroc(pred, label)
        metric_recall = self.metric_recall(pred, label)
        mretric_spec = self.metric_spec(pred, label)
        mretric_acc = self.metric_acc(pred, label)
        mretric_mcc = self.metric_mcc(pred, label)
        precision_tmp, recall_tmp, _ = self.metric_prauc(pred, label)
        metric_prauc = auc(recall_tmp, precision_tmp)
        return metric_auroc, metric_recall, mretric_spec, mretric_acc, mretric_mcc, metric_prauc

    def loss_function1(self, pred, data_label):
        loss = self.criterion(pred, data_label)
        return loss


    def train_step(self, train_dataset, valid_dataset, config_opt):
        dataloader_train = train_dataset.get_data_loaders(self.device, set_type='training', pin_para=False, prefetch_factor= config_opt['prefetch_factor'])
        dataloader_valid = valid_dataset.get_data_loaders(self.device, set_type='valid', pin_para=False, prefetch_factor = config_opt['prefetch_factor'])

        total = sum([param.nelement() for param in self.model.parameters()])
        print("Number of parameter: %.2fM" % (total/1e6))

        # Optimizer
        optimizer1 = torch.optim.Adam(
            self.model.parameters(),weight_decay=float(config_opt['weight_decay']),lr=config_opt['lr']
        )
        warmup_epochs = config_opt['warmup_epochs']
        # scehduler
        scheduler = CosineAnnealingLR(optimizer1, T_max=config_opt['epochs'] - warmup_epochs, eta_min=0.000001)
        scheduler_warmup = LambdaLR(optimizer1, lr_lambda=lambda epoch: epoch/warmup_epochs)

        log_trainLoss = 200
        # path of checkpoints file
        model_checkpoints_folder = os.path.join(self.log_dir, 'checkpoints')
        _save_config_file(model_checkpoints_folder)
        print('path of log file:', self.log_dir)
        log_tem = 0
        patience = 0
        early_stop_patience = config_opt['early_stop_patience']
        best_valid_roc = 0
        for epoch_counter in range(config_opt['epochs']):
            print(datetime.now().strftime('%b%d_%H-%M-%S'))
            if epoch_counter < warmup_epochs:
                scheduler_warmup.step()
            else:
                scheduler.step()

            print('learning rate:', optimizer1.param_groups[0]['lr'])
            self.writer.add_scalar('dtiModel_LR', optimizer1.param_groups[0]['lr'], global_step=epoch_counter)

            for bn_num, data in enumerate(dataloader_train):
                for j in range(2):
                    data[j] = data[j].to(self.device)
                optimizer1.zero_grad()
                pred, cluster_loss, o_loss = self.model(data[0], data[1], config_opt['weight_pocket'])
                pred_loss = self.loss_function1(pred, data[0].data_label.unsqueeze(1).float())
                dti_loss = pred_loss + cluster_loss + config_opt['weight_oloss'] * o_loss

                if log_tem % log_trainLoss == 0:
                    self.writer.add_scalar('pred_loss', pred_loss.item(), global_step=log_tem)
                    self.writer.add_scalar('cluster_loss', cluster_loss.item(), global_step=log_tem)
                    self.writer.add_scalar('o_loss', o_loss.item(), global_step=log_tem)
                log_tem += 1

                dti_loss.backward()
                optimizer1.step()

            _, roc_valid = self.valid_step(dataloader_valid, epoch_counter, config_opt['weight_pocket'])


            if roc_valid > best_valid_roc:
                best_valid_roc = roc_valid
                torch.save(self.model.state_dict(), os.path.join(model_checkpoints_folder, 'model_valid.pth'))

                patience = 0
            else:
                patience += 1
            print('the best valid ROCAUC epoch:', epoch_counter - patience)
            if patience >= early_stop_patience:
                print('the best valid ROCAUC epoch:', best_valid_roc)
                break


    def valid_step(self, valid_loader, epoch_num, weight_pocket):
        with torch.no_grad():
            self.model.eval()
            predictions = []
            labels = []
            label_pred = []
            valid_loss = 0.0
            num_data = 0
            cluster_loss_list = []
            o_loss_list = []
            
            for i, data in enumerate(valid_loader):
                for j in range(2):
                    data[j] = data[j].to(self.device)


                pred, cluster_loss, o_loss = self.model(data[0], data[1], weight_pocket)
                cluster_loss_list.append(cluster_loss.item())
                o_loss_list.append(o_loss.item())

                loss = self.loss_function1(pred, data[0].data_label.unsqueeze(1).float())
                valid_loss += loss.item() * data[0].data_label.size(0)
                num_data += data[0].data_label.size(0)

                predictions.extend(pred.cpu().detach().flatten().numpy())
                labels.extend(data[0].data_label.cpu().flatten().numpy())
            valid_loss /= num_data
        self.model.train()

        predictions = torch.tensor(predictions)
        labels = torch.tensor(labels)
        auroc_score, reacall_score, spec_score, acc_score, mcc_score, metric_prauc = self.compute_metric(predictions, labels)

        print('Epoch:', epoch_num, 'valid loss:', valid_loss, 'cluster_loss:', sum(cluster_loss_list)/len(cluster_loss_list), 'o_loss:', sum(o_loss_list)/len(o_loss_list))
        print('valid ROC AUC:', auroc_score, 'valid Recall:', reacall_score, 'valid Spec:', spec_score, 'valid ACC:', acc_score, 'valid MCC:', mcc_score, 'valid PRAUC:', metric_prauc)    
        return valid_loss, auroc_score

    def load_best_state_dict(self):
        try:
            checkpoints_folder = self.log_dir
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=self.device)
            self.model.load_my_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

def set_random_seed(seed=2024):
    print('random seed：', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    uniprot_df = pd.read_csv('xxxxx/Data/bindingDB/bindingDB_protein_info.csv')

    filename_list = uniprot_df['uniprot_id'].tolist()
    prot_len_list = uniprot_df['len'].tolist()
    name_len_list = []
    for i in range(len(filename_list)):
        name_len_list.append((filename_list[i],prot_len_list[i]))

    with Pool(4)as proc:
        protpocket_list = list(
            tqdm(
                proc.imap(readpocket_cpu, filename_list,
                        ),
                total=len(filename_list)
            ))
    with Pool(4)as proc:
        protfeat_list = list(
            tqdm(
                proc.imap(readfeat_cpu, name_len_list,
                        ),
                total=len(name_len_list)
            ))
    with Pool(4)as proc:
        protedge_list = list(
            tqdm(
                proc.imap(readedge_cpu, filename_list,
                        ),
                total=len(filename_list)
            ))
    with Pool(4)as proc:
        protdegree_list = list(
            tqdm(
                proc.imap(readdegree_cpu, filename_list,
                        ),
                total=len(filename_list)
            ))

    set_random_seed(2024)
    config_init = yaml.load(open("xxxxx/config_init_bindingDB.yaml", "r"), Loader=yaml.FullLoader)
    csv_data_path = 'xxxxx/Data/bindingDB/bindingDB_split_cluster.csv'
    train_dataset = dti_DatasetWrapper(batch_size=config_init['batch_size'], num_workers=config_init['num_worers_train'], data_path=csv_data_path)
    valid_dataset = dti_DatasetWrapper(batch_size=config_init['batch_size'], num_workers=4, data_path=csv_data_path)
    print(config_init)

    dti_predmodel = Model_train(config_init)
    dti_predmodel.train_step(train_dataset, valid_dataset, config_init)






