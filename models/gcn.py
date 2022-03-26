import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp


# GCN based model
class GCNNet(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128,num_features_xd=78, num_features_xt=25, output_dim=128, dropout=0.2):

        super(GCNNet, self).__init__()

        # SMILES graph branch
        self.n_output = n_output
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.fc_g1 = torch.nn.Linear(num_features_xd*4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # protein sequence branch (1d conv)
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.conv_xt_2 = nn.Conv1d(in_channels=32, out_channels=2 * n_filters, kernel_size=8)
        self.conv_xt_3 = nn.Conv1d(in_channels=64, out_channels=3 * n_filters, kernel_size=8)
        self.conv_xt_4 = nn.Conv1d(in_channels=96, out_channels=4 * n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(128 * 33, 1024)
        self.fc2_xt = nn.Linear(1024, 128)
        #self.fc1_xt = nn.Linear(32*121, output_dim)
        '''
         # drugs
        self.embedding_xds = nn.Embedding(num_embeddings=65, embedding_dim=128)
        self.conv_xds_1 = nn.Conv1d(in_channels=100, out_channels=25, kernel_size=8)
        self.conv_xds_2 = nn.Conv1d(in_channels=25, out_channels=50, kernel_size=8)
        self.conv_xds_3 = nn.Conv1d(in_channels=50, out_channels=75, kernel_size=8)
        self.conv_xds_4 = nn.Conv1d(in_channels=75, out_channels=100, kernel_size=8)
        self.fc1_xds = nn.Linear(100 * 50, 1024)
        self.fc2_xds = nn.Linear(1024, 128)

        '''
        # combined layers
        self.fc1 = nn.Linear(2*output_dim, 1024)
        #self.fc1 = nn.Linear(3*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data):
        # get graph input
        x, edge_index, batch = data.x, data.edge_index, data.batch
        '''
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
        '''
        # get protein input
        target = data.target

        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)       # global max pooling

        # flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

        # 1d conv layers
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        conv_xt = torch.relu(conv_xt)
        conv_xt = self.conv_xt_2(conv_xt)
        conv_xt = torch.relu(conv_xt)
        conv_xt = self.conv_xt_3(conv_xt)
        conv_xt = torch.relu(conv_xt)
        conv_xt = self.conv_xt_4(conv_xt)
        conv_xt = torch.relu(conv_xt)
        conv_xt = torch.max_pool1d(conv_xt, kernel_size=3)
        # flatten
        xt = conv_xt.view(-1, 128 * 33)
        xt = torch.relu(self.fc1_xt(xt))
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
