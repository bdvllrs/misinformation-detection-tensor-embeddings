import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import AGNNConv

#dataset_all = ['Pubmed', 'CiteSeer', 'Cora']
dataset = 'Pubmed'

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, T.NormalizeFeatures())
data = dataset[0]


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = torch.nn.Linear(dataset.num_features, 16)
        self.prop1 = AGNNConv(requires_grad=False)
        self.prop2 = AGNNConv(requires_grad=True)
        self.prop3 = AGNNConv(requires_grad=True)
        self.prop4 = AGNNConv(requires_grad=True)
        self.lin2 = torch.nn.Linear(16, dataset.num_classes)

        self.prop1_mat_weight = torch.tensor(0) 
        self.prop2_mat_weight = torch.tensor(0)
        self.prop3_mat_weight = torch.tensor(0)
        self.prop4_mat_weight = torch.tensor(0)

        self.edge_index_with_self_lopp = torch.tensor(0)

    def forward(self):
        x = F.dropout(data.x,0.5,training=self.training)
        x = F.relu(self.lin1(data.x))

        x = self.prop1(x, data.edge_index)
        self.edge_index_with_self_loop , self.prop1_mat_weight = self.prop1.propagation_matrix(x, data.edge_index)

        x = self.prop2(x, data.edge_index)
        self.edge_index_with_self_loop , self.prop2_mat_weight = self.prop2.propagation_matrix(x, data.edge_index)

        x = self.prop3(x, data.edge_index)
        self.edge_index_with_self_loop , self.prop3_mat_weight = self.prop3.propagation_matrix(x, data.edge_index)

        x = self.prop4(x, data.edge_index)
        self.edge_index_with_self_loop , self.prop4_mat_weight = self.prop4.propagation_matrix(x, data.edge_index)

        x = F.dropout(x,0.5,training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.008, weight_decay=0.001)


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


best_val_acc = test_acc = 0

num_epochs = 1000

for epoch in range(1, num_epochs+1):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))

#---------------Get the Propagation Matrix for every Propagation Layer------------------------------------

print('----Edge Index With Self Loop-------')
print(model.edge_index_with_self_loop)

print('----Edge Index With Self Loop shape---')
print(model.edge_index_with_self_loop.shape)


#----Printing the Shapes of the Propagation Matrices for corresponding Propagation Layer

print('Propagation Matrix Weights Shape')

print(model.prop1_mat_weight.shape)
print(model.prop2_mat_weight.shape)
print(model.prop3_mat_weight.shape)
print(model.prop4_mat_weight.shape)


#-----------Printing the Propagation Weights for 3 Propagation Layers ----------------------

print('Propagation Matrix Weight')

print(model.prop1_mat_weight)
print(model.prop2_mat_weight)
print(model.prop3_mat_weight)
print(model.prop4_mat_weight)
