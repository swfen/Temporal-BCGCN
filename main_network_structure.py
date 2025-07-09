from torch.nn import Linear
import torch.nn as nn
from net.Category_Pool_TopK import Category_TopK_Pooling

import torch

from net.TemporalConv import TemporalConv


node_input_dim = node_len
edge_input_dim = rang
retain_node = 10
Tendency = 0.7

class Temporal_BCGCN(torch.nn.Module):  # ECC  alias of NNConv
    def __init__(self, node_input_dim, batch_size):
        super(ECC, self).__init__()
        torch.manual_seed(12)

        self.conv1 = TemporalConv(node_input_dim, node_input_dim, node_num, batch_size)
        self.conv2 = TemporalConv(node_input_dim, node_input_dim, node_num, batch_size)
        self.pool3 = Category_TopK_Pooling(node_input_dim, retain_node, Tendency)

        self.fla = torch.nn.Flatten()
        self.lin = Linear(retain_node * node_len, 2 * retain_node * node_len + 1)
        self.classifier = Linear(2 * retain_node * node_len + 1, 1)
        self.activate = nn.Sigmoid()

    def forward(self, x, edge_index, edge_attr, batch):
        # 输入特征与邻接矩阵（注意格式，上面那种）

        # x = self.lin1(x).relu()
        # dropout = nn.Dropout(p=0.5)
        # x_drop = dropout(x)
        x = self.conv1(x, edge_index, edge_attr)
        bd = nn.BatchNorm1d(node_len, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).to(device)
        x = x.to(torch.float32)
        x = bd(x)
        # (x, edge_index, edge_attr, batch, perm, score) = self.pool0(x, edge_index, edge_attr, batch)
        # dedge_index, dedge_attr = tu.dropout_adj(edge_index, edge_attr, p=0.7)

        # dropout = nn.Dropout(p=0.5)
        # x_drop = dropout(x)
        x = self.conv2(x, edge_index, edge_attr)
        bd = nn.BatchNorm1d(node_len, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).to(device)
        x = bd(x)

        (x, edge_index, edge_attr, batch, perm, perm_lab) = self.pool3(x, edge_index, edge_attr, batch)

        # 分类层
        # x = self.lin(x)
        x = torch.reshape(x, (train_loader.batch_size, retain_node, node_len))
        x = self.fla(x)
        x = self.lin(x)
        out = self.classifier(x).view(-1)
        out = self.activate(out)
        return out, perm


device = torch.device('cuda:0')
model = Temporal_BCGCN(node_input_dim, train_loader.batch_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss().to(device)
