import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp

# GAT  model
class GATNet(torch.nn.Module):
    def __init__(self, num_features_xd=78, n_output=1, num_features_xt=25,
                     n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(GATNet, self).__init__()

        # graph layers
        self.gcn1 = GATConv(num_features_xd, num_features_xd, heads=10)
        self.gcn2 = GATConv(num_features_xd * 10, num_features_xd * 10, heads=2)
        self.gcn3 = GATConv(num_features_xd * 20, num_features_xd * 40)
        self.fc_g1 = nn.Linear(num_features_xd * 40, 1024)
        self.fc_g2 = nn.Linear(1024, 128)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        #self.conv_xt1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        #self.fc_xt1 = nn.Linear(32*121, output_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.conv_xt_2 = nn.Conv1d(in_channels=32, out_channels=2 * n_filters, kernel_size=8)
        self.conv_xt_3 = nn.Conv1d(in_channels=64, out_channels=3 * n_filters, kernel_size=8)
        self.conv_xt_4 = nn.Conv1d(in_channels=96, out_channels=4 * n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(128 * 33, 1024)
        self.fc2_xt = nn.Linear(1024, 128)
        ''' 
        # drugs
        self.embedding_xds = nn.Embedding(num_embeddings=98, embedding_dim=128)
        self.conv_xds_1 = nn.Conv1d(in_channels=100, out_channels=25, kernel_size=8)
        self.conv_xds_2 = nn.Conv1d(in_channels=25, out_channels=50, kernel_size=8)
        self.conv_xds_3 = nn.Conv1d(in_channels=50, out_channels=75, kernel_size=8)
        self.conv_xds_4 = nn.Conv1d(in_channels=75, out_channels=100, kernel_size=8)
        self.fc1_xds = nn.Linear(100 * 50, 1024)
        self.fc2_xds = nn.Linear(1024, 128)
        '''
        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        #self.fc1 = nn.Linear(384, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        '''
        #SMILES
        drugs = data.drugs
        embedded_xds = self.embedding_xds(drugs)
        conv_xds = self.conv_xds_1(embedded_xds)
        conv_xds = torch.relu(conv_xds)
        conv_xds = self.conv_xds_2(conv_xds)
        conv_xds = torch.relu(conv_xds)
        conv_xds = self.conv_xds_3(conv_xds)
        conv_xds = torch.relu(conv_xds)
        conv_xds = self.conv_xds_4(conv_xds)
        conv_xds = torch.relu(conv_xds)
        conv_xds = torch.max_pool1d(conv_xds, kernel_size=2)

        # flatten
        myxds = conv_xds.view(-1, 100 * 50)
        myxds = torch.relu(self.fc1_xds(myxds))
        myxds = self.dropout(myxds)
        myxds = self.fc2_xds(myxds)
        drugs = myxds
        drugs = self.dropout(drugs)
        #drugs = F.dropout(drugs, p=0.2, training=self.training)  # [512,128]
        '''
        # graph input feed-forward
        x, edge_index, batch = data.x, data.edge_index, data.batch

        #x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.gcn1(x, edge_index))
        #x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        x = self.gcn3(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)          # global max pooling
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        #x = self.dropout(x)
        x = F.dropout(x, p=0.2, training=self.training)
        # protein input feed-forward:
        target = data.target
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        conv_xt = torch.relu(conv_xt)
        conv_xt = self.conv_xt_2(conv_xt)
        conv_xt = torch.relu(conv_xt)
        conv_xt = self.conv_xt_3(conv_xt)
        conv_xt = torch.relu(conv_xt)
        conv_xt = self.conv_xt_4(conv_xt)
        conv_xt = torch.relu(conv_xt)
        conv_xt = torch.max_pool1d(conv_xt, kernel_size=3)  # 两层 [512,64,38]

        # flatten
        #print("=========conv_xt.shape=========")
        #print(conv_xt.shape)
        xt = conv_xt.view(-1, 128 * 33)
        xt = torch.relu(self.fc1_xt(xt))# [512,128]
        xt = self.dropout(xt)
        xt = self.fc2_xt(xt)
        xt = self.dropout(xt)
        # concat
        #xc = torch.cat((drugs, x, xt), 1)
        xc = torch.cat((x, xt), 1)

        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
