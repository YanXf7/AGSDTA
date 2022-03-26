import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size, features, edge_index

def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]): 
        x[i] = seq_dict[ch]
    return x  

def seq_cat_sml(sml):
    x = np.zeros(max_seq_len_sml)
    for i, ch in enumerate(sml[:max_seq_len_sml]):
        x[i] = seq_dict_sml[ch]
    return x

# from DeepDTA data
all_prots = []
datasets = ['kiba','davis']
for dataset in datasets:
    print('convert data from DeepDTA for ', dataset)
    fpath = 'data/' + dataset + '/'
    train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))
    train_fold = [ee for e in train_fold for ee in e ]
    valid_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))
    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(fpath + "Y","rb"), encoding='latin1')
    drugs = []
    prots = []
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]),isomericSmiles=True)
        drugs.append(lg)
    for t in proteins.keys():
        prots.append(proteins[t])
    if dataset == 'davis':
        affinity = [-np.log10(y/1e9) for y in affinity]
    affinity = np.asarray(affinity)
    opts = ['train','test']
    for opt in opts:
        rows, cols = np.where(np.isnan(affinity)==False)  
        if opt=='train':
            rows,cols = rows[train_fold], cols[train_fold]
        elif opt=='test':
            rows,cols = rows[valid_fold], cols[valid_fold]
        with open('data/' + dataset + '_' + opt + '.csv', 'w') as f:
            f.write('compound_iso_smiles,target_sequence,affinity\n')
            for pair_ind in range(len(rows)):
                ls = []
                ls += [ drugs[rows[pair_ind]]  ]
                ls += [ prots[cols[pair_ind]]  ]
                ls += [ affinity[rows[pair_ind],cols[pair_ind]]  ]
                f.write(','.join(map(str,ls)) + '\n')       
    print('\ndataset:', dataset)
    print('train_fold:', len(train_fold))
    print('test_fold:', len(valid_fold))
    print('len(set(drugs)),len(set(prots)):', len(set(drugs)),len(set(prots)))
    all_prots += list(set(prots))

seq_voc_sml = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
				"1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
				"9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
				"D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
				"O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
				"V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
				"b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
				"l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

#seq_voc_sml = 'CNOSFSiPClBrMgNaCaFeAsAlIBVKTlYbSbSnAgPdCoSeTiZnHLiGeCuAuNiCdInMnZrCrPtHgPbcnh1234567890-=#()[]+.'
seq_dict_sml = {v:(i+1) for i,v in enumerate(seq_voc_sml)}
seq_dict_len_sml = len(seq_dict_sml)
max_seq_len_sml = 100
'''
seq_voc = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
				"U": 19, "T": 20, "W": 21,
				"V": 22, "Y": 23, "X": 24,
				"Z": 25 }
'''
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000

compound_iso_smiles = []
for dt_name in ['kiba','davis']:
    opts = ['train','test']
    for opt in opts:
        df = pd.read_csv('data/' + dt_name + '_' + opt + '.csv')
        compound_iso_smiles += list( df['compound_iso_smiles'] )
compound_iso_smiles = set(compound_iso_smiles)
smile_graph = {}
for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    smile_graph[smile] = g

datasets = ['davis','kiba']
# convert to PyTorch data format
for dataset in datasets:
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        df = pd.read_csv('data/' + dataset + '_train.csv')
        train_drugs, train_prots,  train_Y = list(df['compound_iso_smiles']),list(df['target_sequence']),list(df['affinity'])
        XT = [seq_cat(t) for t in train_prots]
        # XDS = [seq_cat(t) for t in train_drugs]
        XDS = [seq_cat_sml(t) for t in train_drugs]
        train_drugs, train_drugs_sml,train_prots,  train_Y = np.asarray(train_drugs), np.asarray(XDS), np.asarray(XT), np.asarray(train_Y)
        df = pd.read_csv('data/' + dataset + '_test.csv')
        test_drugs, test_prots,  test_Y = list(df['compound_iso_smiles']),list(df['target_sequence']),list(df['affinity'])
        XT = [seq_cat(t) for t in test_prots]
        # XDS = [seq_cat(t) for t in test_drugs]
        XDS = [seq_cat_sml(t) for t in test_drugs]
        test_drugs, test_drugs_sml, test_prots,  test_Y = np.asarray(test_drugs), np.asarray(XDS), np.asarray(XT), np.asarray(test_Y)

        # make data PyTorch Geometric ready
        print('preparing ', dataset + '_train.pt in pytorch format!')
        train_data = TestbedDataset(root='data', dataset=dataset+'_train', xd=train_drugs, xds=train_drugs_sml, xt=train_prots, y=train_Y,smile_graph=smile_graph)
        print('preparing ', dataset + '_test.pt in pytorch format!')
        test_data = TestbedDataset(root='data', dataset=dataset+'_test', xd=test_drugs, xds=test_drugs_sml, xt=test_prots, y=test_Y,smile_graph=smile_graph)
        print(processed_data_file_train, ' and ', processed_data_file_test, ' have been created')        
    else:
        print(processed_data_file_train, ' and ', processed_data_file_test, ' are already created')        
