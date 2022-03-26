import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

# GCN-CNN based model

class GAT_GCN(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(GAT_GCN, self).__init__()

        self.n_output = n_output
        self.conv1 = GATConv(num_features_xd, num_features_xd, heads=10)
        self.conv2 = GATConv(num_features_xd * 10, num_features_xd * 10, heads=2)
        #self.conv2 = GCNConv(num_features_xd*10, num_features_xd*10)
        self.conv3 = GCNConv(num_features_xd*20, num_features_xd*40)
        self.fc_g1 = torch.nn.Linear(num_features_xd*40*2, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.conv_xt_2 = nn.Conv1d(in_channels=32, out_channels=2 * n_filters, kernel_size=8)
        self.conv_xt_3 = nn.Conv1d(in_channels=64, out_channels=3 * n_filters, kernel_size=8)
        self.conv_xt_4 = nn.Conv1d(in_channels=96, out_channels=4 * n_filters, kernel_size=8)
        #self.conv_xt_5 = nn.Conv1d(in_channels=128, out_channels=5 * n_filters, kernel_size=8)
      
        self.fc1_xt = nn.Linear(128 * 33, 1024)
        self.fc2_xt = nn.Linear(1024, 128)
        
        # drugs
        self.embedding_xds = nn.Embedding(num_embeddings=65, embedding_dim=128)
        self.conv_xds_1 = nn.Conv1d(in_channels=100, out_channels=25, kernel_size=8)
        self.conv_xds_2 = nn.Conv1d(in_channels=25, out_channels=50, kernel_size=8)
        self.conv_xds_3 = nn.Conv1d(in_channels=50, out_channels=75, kernel_size=8)
        self.conv_xds_4 = nn.Conv1d(in_channels=75, out_channels=100, kernel_size=8)
        self.fc1_xds = nn.Linear(100 * 50, 1024)
        self.fc2_xds = nn.Linear(1024, 128)
    
        # combined layers
        self.fc1 = nn.Linear(384, 1024)
        #self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)        # n_output = 1 for regression task

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target
    
        drugs = data.drugs
        #drugs
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
        # drugs = F.dropout(drugs, p=0.2, training=self.training)  #[512,128]
        drugs = self.dropout(drugs)

        # print('x shape = ', x.shape)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.conv3(x, edge_index)
        x = self.relu(x)

        # apply global max pooling (gmp) and global mean pooling (gap)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        #print("======x.shape======")
        #print(x.shape)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = F.dropout(x, p=0.2, training=self.training)
    
        #Protein
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        conv_xt = torch.relu(conv_xt)
        conv_xt = self.conv_xt_2(conv_xt)
        conv_xt = torch.relu(conv_xt)
        conv_xt = self.conv_xt_3(conv_xt)
        conv_xt = torch.relu(conv_xt)
        conv_xt = self.conv_xt_4(conv_xt)
        conv_xt = torch.relu(conv_xt)
        conv_xt = torch.max_pool1d(conv_xt, kernel_size=3)   # 两层 [512,64,38]
        # flatten
        xt = conv_xt.view(-1, 128 * 33)
        xt = torch.relu(self.fc1_xt(xt))#[512,128]
        xt = self.dropout(xt)
        xt = self.fc2_xt(xt)
        xt = self.dropout(xt)

        # concat
        xc = torch.cat((drugs, x, xt), 1)
        #xc = torch.cat((drugs, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
