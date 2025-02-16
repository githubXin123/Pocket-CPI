import torch
import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader

from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem import rdPartialCharges
import numpy as np

permitted_list_of_atoms = [
            "B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S", "Si", 'Se', 'Te'
        ]

permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE,
                                    Chem.rdchem.BondType.DOUBLE,
                                    Chem.rdchem.BondType.TRIPLE,
                                    Chem.rdchem.BondType.AROMATIC]

def one_hot_encoding(x, permitted_list):
    """
    Creates a binary one-hot encoding of x with respect to the elements in permitted_list. Identifies an input element x that is not in permitted_list with the last element of permitted_list.
    """

    if x not in permitted_list:
        x = permitted_list[-1]

    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]

    return binary_encoding

def sigmoid(number: float):
    """ numerically semi-stable sigmoid function to map charge between 0 and 1 """
    return 1.0 / (1.0 + float(np.exp(-number)))

def get_feature(para_tem):
    smi = para_tem[0]
    smi_name = para_tem[1]
    mol = Chem.MolFromSmiles(smi)

    xs = []

    for atom in mol.GetAtoms():
        # atom typ
        atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
        # degree
        n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
        # formal charge
        formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
        #  hybridization type
        hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
        
        # is in a ring
        is_in_a_ring_enc = [int(atom.IsInRing())]
        # is aromatic
        is_aromatic_enc = [int(atom.GetIsAromatic())]
        # atomic mass
        atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]
        # vdw radius
        vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]
        # covalent radius
        covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]

        # chirality type
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        # n_hydrogens
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])

        atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc \
                                + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc \
                                + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled \
                                + chirality_type_enc + n_hydrogens_enc

        x = torch.tensor(atom_feature_vector)
        xs.append(x)

    mol_x = torch.stack(xs, dim=0)

    edge_indices = []
    edge_attrs = []

    for bond in mol.GetBonds():
        edge_indices += [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]]

        permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE,
                                    Chem.rdchem.BondType.DOUBLE,
                                    Chem.rdchem.BondType.TRIPLE,
                                    Chem.rdchem.BondType.AROMATIC]

        bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
        bond_is_conj_enc = [int(bond.GetIsConjugated())]
        bond_is_in_ring_enc = [int(bond.IsInRing())]
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])

        bond_feature = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc + stereo_type_enc

        edge_attrs.append(bond_feature)
    edge_attrs = torch.tensor(edge_attrs, dtype=torch.float)
    mol_edge_attr = torch.cat([edge_attrs, edge_attrs], dim = 0)

    edge_indices = torch.tensor(edge_indices, dtype=torch.long).view(-1,2)
    edge_indices = torch.cat([edge_indices, edge_indices[:, [1, 0]]], dim=0)

    mol_edge_index = edge_indices.t().contiguous()

    torch.save(mol_x, "/xxxxx/processing data/bindingDB/mol_info/node_attr/" + str(smi_name) + ".pt")
    torch.save(mol_edge_index, "/xxxxx/processing data/bindingDB/mol_info/edge_index/" + str(smi_name) + ".pt")
    torch.save(mol_edge_attr, "/xxxxx/processing data/bindingDB/mol_info/edge_attr/" + str(smi_name) + ".pt")
    return 1


df_mol = pd.read_csv('/xxxxx/Data/bindingDB/our_code/bindingDB/bindingDB_mol_info.csv')
molname_list = df_mol['mol_id'].tolist()
smiles_list = df_mol['smiles'].tolist()


para_tem = []
for i in range(len(molname_list)):
    para_tem.append((smiles_list[i], molname_list[i]))

# para_tem = []
# mol_name = 'cs_mol_5'
# mol_smiles = 'CC[C@H]1CN(c2c(cccn2)O1)Cc3ccc(cc3)c4ccccc4c5[nH]nnn5'
# para_tem.append((mol_smiles, mol_name))

with Pool(20)as proc:
    protfeat_list = list(
        tqdm(
            proc.imap(get_feature, para_tem,
                    ),
            total=len(para_tem)
        ))
