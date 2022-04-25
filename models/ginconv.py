import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool, global_max_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from models.ExternalAttention import ExternalAttention

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# GINConv model
class GINConvNet(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(GINConvNet, self).__init__()

        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        
        # convolution layers
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)
        
        self.fc1_xd = Linear(dim, output_dim)
    
        #self.fc2_xd = Linear(1024, output_dim)
        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.conv_xt_2 = nn.Conv1d(in_channels=32, out_channels=2 * n_filters, kernel_size=8)
        self.conv_xt_3 = nn.Conv1d(in_channels=64, out_channels=3 * n_filters, kernel_size=8)
        self.conv_xt_4 = nn.Conv1d(in_channels=96, out_channels=4 * n_filters, kernel_size=8)

        self.fc1_xt = nn.Linear(128 * 33, 1024)
        self.fc2_xt = nn.Linear(1024, 128)
        #combined layers

        self.fc1 = nn.Linear(384, 1024)
        #self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)        # n_output = 1 for regression task

    
        # drugs
        self.embedding_xds = nn.Embedding(num_embeddings=65, embedding_dim=128)
        self.conv_xds_1 = nn.Conv1d(in_channels=100, out_channels=25,kernel_size=8)
        self.conv_xds_2 = nn.Conv1d(in_channels=25, out_channels=50, kernel_size=8)
        self.conv_xds_3 = nn.Conv1d(in_channels=50, out_channels=75, kernel_size=8)
        self.conv_xds_4 = nn.Conv1d(in_channels=75, out_channels=100, kernel_size=8)
        self.fc1_xds = nn.Linear(100 * 50, 1024)
        self.fc2_xds = nn.Linear(1024, 128)
        
        self.attention_layer = ExternalAttention(d_model=128,S=4)

        self.graph_attention_layer = nn.Linear(128, 128)
        self.drug_attention_layer = nn.Linear(128, 128)





    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target

        drug = data.drugs
        embedded_xds = self.embedding_xds(drug)

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

    
        # Graph
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x  = self.bn5(x)
        x = global_add_pool(x, batch)   # [512,32]

        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)
    


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
        

        drugs_a = torch.unsqueeze(drugs, 2)
        x_a = torch.unsqueeze(x, 2)
        
        drugs_att = self.drug_attention_layer(drugs_a.permute(0, 2, 1))
        x_att = self.graph_attention_layer(x_a.permute(0, 2, 1))
        
        drugs_att_layers = torch.unsqueeze(drugs_att, 2).repeat(1, 1, x_a.shape[-1], 1)  # repeat along graph size
        x_att_layers = torch.unsqueeze(x_att, 1).repeat(1, drugs_a.shape[-1], 1, 1)  # repeat along drug size
        
        Atten_matrix = self.attention_layer(torch.relu(drugs_att_layers + x_att_layers))
        drugs_atte = torch.mean(Atten_matrix, 2)
        x_atte = torch.mean(Atten_matrix, 1)
        
        drugs_atte = torch.sigmoid(drugs_atte.permute(0, 2, 1))
        x_atte = torch.sigmoid(x_atte.permute(0, 2, 1))

        drugs_atte = torch.squeeze(drugs_atte, 2)
        x_atte = torch.squeeze(x_atte, 2)

        drugs = drugs * 0.5 + drugs * drugs_atte
        x = x * 0.5 + x * x_atte
        

        # concat
        xc = torch.cat((drugs, x, xt), 1)  # [512,256]
        #xc = torch.cat((x, xt), 1)    #[512,256]
        # print("====xc=====")
        # print(xc.shape)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
