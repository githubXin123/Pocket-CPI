'''
computes the degree of each amino acid node in a pdb file, and sets the degree of low-degree nodes to 0
'''

from multiprocessing import Pool
from tqdm import tqdm
import torch
import numpy as np
import os


def compute_contact_map(file_name):

    try:
        path_share = 'xxxxx/pdb_file/'
        f = open(path_share + file_name + '.pdb',"r")
        D = []
        i = 0
        for a in f.readlines():
            b = a.split()
            if b[0] == "ATOM":
                if b[2] == "CA":
                    D.append((float(b[6]),float(b[7]),float(b[8])))
    except:
        print(file_name)


    dis1 = []
    if len(D) > 1022:
        D = D[:1022]

    for b in range(len(D)):
        dis2 = []
        for c in range(len(D)):
            dis = ((D[b][0] - D[c][0]) ** 2 + (D[b][1] - D[c][1]) ** 2 + (D[b][2] - D[c][2]) ** 2) ** 0.5
            if dis == 0.:
                dis2.append(0)
            elif (dis <= 10.) & (dis != 0.):
                dis2.append(1)
            elif dis > 10.:
                dis2.append(0)
        dis1.append(dis2)
    
    a = np.array(dis1)
    edge_index = torch.from_numpy(a)
    edge_index = edge_index.sum(dim=1)

    prot_degree = torch.where(edge_index < 10, 0, edge_index)
    prot_degree = torch.where(prot_degree != 0, 1, prot_degree)
    prot_degree = prot_degree.unsqueeze(dim=1)
    prot_degree = torch.save(prot_degree, '/xxxxx/processing data/bindingDB/prot_info/prot_degree/' + str(file_name) + '.pt')
    return 1


folder_path = 'xxxxx/pdb_fil'
file_names_list = [os.path.splitext(f)[0] for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

with Pool(20)as proc:
    protpocket_list = list(tqdm(
            proc.imap(compute_contact_map, file_names_list),
            total=len(file_names_list)
        ))

print('finish')









