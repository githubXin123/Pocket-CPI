'''
According to the pdb file, compute contact map
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
                dis2.append(round(float(dis),3))
            elif dis > 10.:
                dis2.append(0)
        dis1.append(dis2)
    
    a = np.array(dis1)
    edge_index = torch.from_numpy(a)
    edge_index = edge_index.nonzero(as_tuple=False).t().contiguous()
    edge_index = torch.save(edge_index, '/xxxxx/processing data/bindingDB/prot_info/contact_map/' + str(file_name) + '.pt')
    return 1



folder_path = 'xxxxx/pdb_file'
file_names_list = [os.path.splitext(f)[0] for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

with Pool(20)as proc:
    protpocket_list = list(tqdm(
            proc.imap(compute_contact_map, file_names_list),
            total=len(file_names_list)
        ))

print('finish')









