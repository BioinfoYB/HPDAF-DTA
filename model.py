import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch.nn.modules.transformer import _get_clones, _get_activation_fn
from torch_geometric.nn import GINConv, global_add_pool, global_max_pool,GATConv,global_mean_pool,GCNConv,SAGEConv
import numpy as np


d_model=100
dim_feedforward = 256
n_heads = 2
n_layers=2

def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table)

class B2CN(nn.Module):
    def __init__(self, input_size=100, hidden_size=256, num_layers=2, dropout=0.1, kernel_size=3, out_channels=64):
        super(B2CN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=out_channels, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.bilstm = nn.LSTM(out_channels, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        
        self.fc = nn.Linear(hidden_size * 2, 100)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.bilstm(x)  
        lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out[:, -1, :]
        output = self.fc(lstm_out)

        return output
    
class GIN(torch.nn.Module):
    def __init__(self, c_feature=108,MLP_dim=96):
        super(GIN, self).__init__()
        nn1 = Sequential(Linear(c_feature, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(MLP_dim)
        nn2 = Sequential(Linear(MLP_dim, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(MLP_dim)
        nn3 = Sequential(Linear(MLP_dim, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(MLP_dim)

        self.lin = Linear(MLP_dim, 120)
        
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()

        x = global_add_pool(x, batch)

        x = F.dropout(x, p=0.1)
        x = self.lin(x)

        return x

class SAGE(torch.nn.Module):
    def __init__(self,c_feature=108,MLP_dim=88):
        super().__init__()
        self.conv1 = SAGEConv(c_feature,MLP_dim,aggr='mean')
        self.conv2 = SAGEConv(MLP_dim, MLP_dim, aggr='mean')
        self.conv3 = SAGEConv(MLP_dim, MLP_dim, aggr='mean')
        self.lin = Linear(MLP_dim,100)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = global_add_pool(x, batch)
        x = self.lin(x)

        return x

class MACNet(nn.Module):
    def __init__(self, input_channels=3, attention_dim=100):
        super(MACNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, input_channels, kernel_size=3, padding=1)
        
        self.attn_fc = nn.Linear(10*10, attention_dim)
        self.attn_weights = nn.Parameter(torch.ones(1, attention_dim))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        batch_size, C, H, W = x.size()
        x_flattened = x.view(batch_size * C, H * W)
        
        attn_scores = torch.sigmoid(self.attn_fc(x_flattened))
        attn_scores = attn_scores.view(batch_size, C, H, W)
        
        weighted_x = x * attn_scores
        
        return weighted_x, attn_scores
    
class AACNet(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(AACNet, self).__init__()
        
        self.attn_fc = nn.Linear(input_dim, attention_dim)
        
        self.attn_weights = nn.Parameter(torch.ones(1, attention_dim))

    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        x_flattened = x.view(batch_size * C, H * W)

        attn_scores = torch.sigmoid(self.attn_fc(x_flattened))
        attn_scores = attn_scores.view(batch_size, C, H, W)

        weighted_x = x * attn_scores

        return weighted_x, attn_scores

class HPDAF_DTA(torch.nn.Module):
    def __init__(self, MLP_dim=82, dropout=0.1, c_feature=108):
        super(HPDAF_DTA, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.attention_weights = nn.Parameter(torch.Tensor(3, 300))
        nn.init.xavier_uniform_(self.attention_weights)

        self.src_emb = nn.Embedding(26, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(26, d_model), freeze=True)

        self.protein_seq = B2CN(input_size=100, hidden_size=256, num_layers=2, dropout=dropout)
        self.drug_graph = GIN()
        self.complex_graph = SAGE()

        self.attention_layer = MACNet(input_channels=3, attention_dim=100)
        self.self_attention_layer = AACNet(input_dim=100, attention_dim=100)


        self.attention_fc = nn.Linear(100 + 100 + 100, 300)
        self.attention_weights = nn.Parameter(torch.Tensor(3, 300))
        nn.init.xavier_uniform_(self.attention_weights)

        self.fc1_c = Linear(MLP_dim, 100)
        self.poc_fc = Linear(100, 60)

        self.classifier = nn.Sequential(
            nn.Linear(100 + 100 + 100, 512),
            nn.Dropout(0.1),
            nn.PReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.1),
            nn.PReLU(),
            nn.Linear(256, 1),
        )
        
       
    def forward(self, data):
        x_ligand, edge_index_ligand, batch_ligand = data.x_t, data.edge_index_t, data.x_t_batch
        x_complex, edge_index_complex, batch_complex = data.x_s, data.edge_index_s, data.x_s_batch
        protein = data.protein

        x_ligand = self.drug_graph(x_ligand, edge_index_ligand, batch_ligand)

        com = self.complex_graph(x_complex, edge_index_complex, batch_complex)

        protein = self.src_emb(protein) + self.pos_emb(protein)
        protein = self.protein_seq(protein)

        protein = protein.view(protein.shape[0], 10, 10)
        com = com.view(com.shape[0], 10, 10)
        x_ligand = x_ligand.view(x_ligand.shape[0], 10, 10)

        stacked = torch.stack([protein, com, x_ligand], dim=1)

        weighted_stacked, attn_scores = self.attention_layer(stacked)
        weighted_stacked, attn_scores_self = self.self_attention_layer(weighted_stacked)


        weighted_stacked = weighted_stacked.view(weighted_stacked.shape[0], -1)

        combined_features = self.attention_fc(weighted_stacked)
        attention_scores = torch.matmul(combined_features, self.attention_weights.T)

        attention_weights = F.softmax(attention_scores, dim=-1)

        x_split = torch.split(combined_features, [100, 100, 100], dim=1)

        weighted_features = torch.zeros_like(x_split[0])
        for i, part in enumerate(x_split):
            weighted_features += (attention_weights[:, i].unsqueeze(1) * part)

        x = self.classifier(weighted_stacked)

        return x

