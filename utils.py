import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis', 
                 xd=None, xds=None, xt=None, y=None, transform=None,
                 pre_transform=None,smile_graph=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        # dim = 32
        # dropout = 0.2
        # n_output = 1
        # output_dim = 128
        #
        # self.dropout = nn.Dropout(dropout)
        # self.relu = nn.ReLU()
        # self.n_output = n_output
        # # convolution layers
        # nn1 = Sequential(Linear(98, dim), ReLU(), Linear(dim, dim))
        # self.conv1 = GINConv(nn1)
        # self.bn1 = torch.nn.BatchNorm1d(dim)
        #
        # nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        # self.conv2 = GINConv(nn2)
        # self.bn2 = torch.nn.BatchNorm1d(dim)
        #
        # nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        # self.conv3 = GINConv(nn3)
        # self.bn3 = torch.nn.BatchNorm1d(dim)
        #
        # nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        # self.conv4 = GINConv(nn4)
        # self.bn4 = torch.nn.BatchNorm1d(dim)
        #
        # nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        # self.conv5 = GINConv(nn5)
        # self.bn5 = torch.nn.BatchNorm1d(dim)
        #
        # self.fc1_xd = Linear(dim, output_dim)
        # 打开注释
        # self.embedding_xt = nn.Embedding(num_embeddings=65, embedding_dim=128)
        # self.conv_xt_1 = nn.Conv1d(in_channels=100, out_channels=32, kernel_size=8)
        # self.conv_xt_2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8)
        # self.conv_xt_3 = nn.Conv1d(in_channels=64, out_channels=96, kernel_size=8)
        #
        # self.fc1_xt = nn.Linear(96*53, 128)
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xds, xt, y,smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, xds, xt, y,smile_graph):
        assert (len(xd) == len(xds) and len(xds) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            # 打开注释
            # tmp_tensor = torch.LongTensor(xds[i])
            # tmp_tensor = torch.unsqueeze(tmp_tensor, 0)
            # # print("========tmp_tensor===========")
            # # print(tmp_tensor.shape)  #[100]  --->  [1, 100]
            # embedded_xds = self.embedding_xt(tmp_tensor)
            # conv_xds = self.conv_xt_1(embedded_xds)
            # conv_xds = torch.relu(conv_xds)
            # conv_xds = self.conv_xt_2(conv_xds)
            # conv_xds = torch.relu(conv_xds)
            # conv_xds = self.conv_xt_3(conv_xds)
            # conv_xds = torch.relu(conv_xds)
            # # print("=====conv_xds=====")
            # # print(conv_xds.shape)
            # conv_xds = torch.max_pool1d(conv_xds, kernel_size=2)
            # # print("=====conv_xds=====")
            # # print(conv_xds.shape)
            # # flatten
            # myxds = conv_xds.view(-1, 96 * 53)
            # myxds = self.fc1_xt(myxds)
            # drugs = myxds
            target = xt[i]   #(1000,)
            # print("=======target====")
            # print(target.shape)
            mydrugs = xds[i]
            # print("=======mydrugs====")
            # print(mydrugs.shape)
            labels = y[i]  # ()

            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]))
            GCNData.drugs = torch.LongTensor([mydrugs])
            GCNData.target = torch.LongTensor([target])  # torch.Size([1, 1000])

            # GCNData.drugs = torch.LongTensor([smiles])
            # GCNData.drugs = drugs

            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci