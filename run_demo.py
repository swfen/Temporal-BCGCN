
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

import numpy as np
import torch
from PSA import PSA

from net.TemporalConv import TemporalConv
from sklearn.metrics import roc_auc_score


node_num = 90 #The number of brain regions
w = 115   # window_width
s = 10    # window_Step
y_Lab = np.load('shuffled_dianke_binary.npy')   #  label [0 1]

subjects_timeseries_with_confounds = PSA(w,s,node_num,'dianke')     #node_feature

# subjects_timeseries_with_confounds is a three-dimensional array (sample size, number of sliding windows, number of brain regions).
# For each sample, there are average node signals from 90 brain regions under each sliding window,
# which are stacked up in chronological order to form all the node features for a sample.


dy_connectivity_matrices_OSCC = np.load(str(w)+'_'+str(s)+'dianke_dy_connectivity_matrices.npy') #raw_edge_feature

# dy_connectivity_matrices_OSCC is a three-dimensional array (sample size, number of brain regions × number of sliding Windows, number of brain regions).
# For each sample, there is a functional connection matrix under each sliding window.
# Stacking them in chronological order constitutes all the edge features of a sample.


node_len = subjects_timeseries_with_confounds.shape[1]
rang = len(dy_connectivity_matrices_OSCC[0][:,0]) / node_num
rang = int(rang)                                            #  Number of Windows

class Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['115_10_dianke.dataset']

    def download(self):
        pass

    def process(self):

        edge_index = torch.tensor([[0] * node_num * (node_num - 1)] * 2, dtype=torch.long)
        t = 0
        for i in range(0, node_num):
            r = 0
            for j in range(0, node_num - 1):
                if j == node_num - 1 and i == node_num - 1:
                    break
                if j == i:
                    edge_index[0][j + t * (node_num - 1)] = i
                    edge_index[1][j + t * (node_num - 1)] = j + 1
                    r = 1
                if r == 1:
                    edge_index[0][j + t * (node_num - 1)] = i
                    edge_index[1][j + t * (node_num - 1)] = j + 1
                else:
                    edge_index[0][j + t * (node_num - 1)] = i
                    edge_index[1][j + t * (node_num - 1)] = j

            t = t + 1
        data_list = []

        for i in range(0, len(y_Lab)):

            x = torch.tensor(np.transpose(subjects_timeseries_with_confounds[i]), dtype=torch.float64)
            y = torch.tensor(y_Lab[i], dtype=torch.long)

            edge_att = torch.tensor([[0] * rang] * node_num * (node_num - 1), dtype=torch.float64)
            edge_inde = torch.tensor(edge_index, dtype=torch.long)  # 强制转换为tensor

            for h in range(rang):
                r = 0
                for j in range(node_num):
                    for t in range(node_num):
                        if dy_connectivity_matrices_OSCC[i][node_num * h + t][j] != 1:
                            edge_att[r, h] = dy_connectivity_matrices_OSCC[i][node_num * h + t][j]
                            r = r + 1

            edge_att = torch.tensor(edge_att, dtype=torch.float)  # 强制转换为tensor

            data = Data(x=x, y=y, edge_index=edge_inde, edge_attr=edge_att)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

dataset = Dataset(root='data/')
print()
print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.

print()
print(data)
print('=============================================================')

# Gather some statistics about the first graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')


from torch_geometric.loader import DataLoader


test_dataset = dataset[0:40]
train_dataset = dataset[40:200]


print(f'Number of training graphs: {len(train_dataset)}')
train_loader = DataLoader(train_dataset, batch_size = 10, shuffle=True)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()


print(f'Number of testing graphs: {len(test_dataset)}')
test_loader = DataLoader(test_dataset, batch_size = 10, shuffle=True)

from torch.nn import Linear
import torch.nn as nn
from net.Category_Pool_TopK import Category_TopK_Pooling

node_input_dim = node_len
edge_input_dim = rang
retain_node = 10
Tendency = 0.1


class Temporal_BCGCN(torch.nn.Module):
    def __init__(self, node_input_dim,batch_size):
        super(Temporal_BCGCN, self).__init__()
        torch.manual_seed(12)

        self.conv1 = TemporalConv(node_input_dim, node_input_dim, node_num, batch_size)
        self.conv2 = TemporalConv(node_input_dim, node_input_dim, node_num, batch_size)
        self.pool3 = Category_TopK_Pooling(node_input_dim, retain_node,Tendency)

        self.fla = torch.nn.Flatten()
        self.lin = Linear(node_len,2*node_len+1)
        self.classifier = Linear(2*node_len+1,1)
        self.activate = nn.Sigmoid()

    def forward(self, x, edge_index, edge_attr, batch):

        x = self.conv1(x, edge_index, edge_attr)
        bd = nn.BatchNorm1d(node_len, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).to(device)
        x = x.to(torch.float32)
        x = bd(x)

        x = self.conv2(x, edge_index, edge_attr)
        bd = nn.BatchNorm1d(node_len, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).to(device)
        x = bd(x)

        (x, edge_index, edge_attr, batch, perm, perm_lab) = self.pool3(x, edge_index, edge_attr, batch)


        x = torch.reshape(x, (train_loader.batch_size, retain_node, node_len))
        x = self.activate(torch.mean(x,dim=1))
        x = self.lin(x)
        out = self.classifier(x).view(-1)
        out = self.activate(out)
        return out, perm


device = torch.device('cuda:0')
model = Temporal_BCGCN(node_input_dim,train_loader.batch_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss().to(device)


def train(Loss=[]):
    model.train().to(device)
    # lo = []
    train_correct = 0
    for data in train_loader:  # Iterate in batches over the training dataset.
        data.to(device)
        out, x_perm_lab = model(data.x, data.edge_index, data.edge_attr,
                                data.batch)  # Perform a single forward pass.
        pred = torch.where(out < 0.5, torch.tensor(0).to(device), torch.tensor(1).to(device))
        train_correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        loss = criterion(out, data.y.float()).to(device)  # Compute the loss.
        loss.backward()  # Derive gradients.
        # lo.append(loss)
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
    # L = sum(lo)/len(lo)
    # Loss.append(L)
    # return Loss
    return train_correct / len(train_loader.dataset)


def test(train_loader, test_loader, Loss=[]):
    model.eval().to(device)
    test_correct = 0
    y_true = []
    y_pred = []
    TP = 0  # TP 病人
    TN = 0  # TN 正常人
    TP_FN = 0  # TP+FN 真实病人个数
    TN_FP = 0  # TN+FP 真实正常人个数
    TP_FP = 0  # TP+FP 被判断为病人的个数
    lo = []
    for data in test_loader:  # Iterate in batches over the training/test dataset.
        data.to(device)
        out,perm = model(data.x, data.edge_index, data.edge_attr, data.batch)
        pred = torch.where(out < 0.5, torch.tensor(0).to(device), torch.tensor(1).to(device))
        # loss = criterion(out, data.y).to(device)
        # lo.append(loss)
        test_correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        Len = len(data.y)
        y = data.y.cpu().tolist()
        TP_FN += int(y.count(1))
        TN_FP += int(y.count(0))
        y_true.extend(data.y.tolist())
        y_pred.extend(pred.tolist())

        for i in range(Len):
            if pred[i] == y[i] and pred[i] == 1:
                TP += 1
        for i in range(Len):
            if pred[i] == y[i] and pred[i] == 0:
                TN += 1
        for i in range(Len):
            if pred[i] == 1:
                TP_FP += 1
    # L = sum(lo) / len(lo)
    # Loss.append(L)
    if TP_FP == 0:
        pre = 0
    else:
        pre = TP / TP_FP

    if TN_FP == 0:
        TN_FP = 10000
    if TP_FN == 0:
        TP_FN = 10000
    spe = TN / TN_FP
    sen = TP / TP_FN
    if np.all(np.array(y_pred) == 0):
        y_pred[0] = 1  # 将第一个元素设为1
    if np.all(np.array(y_pred) == 1):
        y_pred[0] = 0  # 将第一个元素设为1
    auroc = roc_auc_score(y_true, y_pred)
    return out, perm, test_correct / len(
        test_loader.dataset), sen, spe, pre, auroc  # Derive ratio of correct predictions.


for epoch in range(1, 501):
    train_acc = train()

    out, x_perm_lab, test_acc, sen, spe, pre, auroc = test(train_loader, test_loader)

    if sen + pre == 0:
        F1_score = 0
    else:
        F1_score = (2 * sen * pre) / (sen + pre)
    print(
        f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, SEN:{sen:.4f},SPE:{spe:.4f},F1_Score:{F1_score:.4f},auroc:{auroc:.4f}')

