#from utils import accuracy
from utils import Config
import numpy as np
from pygcn.utils import accuracy, encode_onehot, normalize, sparse_mx_to_torch_sparse_tensor
from pyagnn.agnn.model import AGNN
from pygcn.models import GCN
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp
import torch

config = Config('config/')
device = torch.device("cuda" if config.learning.cuda else "cpu")


class TrainerGraph:

    def __init__(self, C_nodes, graph, all_labels_init, labels_init):
        self.loss_min = 100
        self.max_acc = 0
        self.epochs = config.learning.epochs
        self.adj = sp.coo_matrix(graph, dtype=np.float32)
        self.all_labels = encode_onehot(all_labels_init)
        self.features = normalize(np.array(C_nodes))
        self.adj = normalize(self.adj + sp.eye(self.adj.shape[0]))
        self.features = torch.FloatTensor(np.array(self.features))
        self.all_labels = torch.LongTensor(np.where(self.all_labels)[1])
        self.adj = sparse_mx_to_torch_sparse_tensor(self.adj)
        self.idx_test = np.where(np.array(labels_init) == 0)[0]
        self.labels = encode_onehot(labels_init)
        self.labels = torch.LongTensor(np.where(self.labels)[1])
        self.idx_train_all = np.where(self.labels)[0]
        print(self.idx_train_all )
        self.all_labels_init = torch.LongTensor(self.all_labels)

        self.idx_train = torch.LongTensor(
            self.idx_train_all[:int((1 - config.learning.ratio_val) * len(self.idx_train_all))])
        self.idx_val = torch.LongTensor(
            self.idx_train_all[int((1 - config.learning.ratio_val) * len(self.idx_train_all)):])

        self.idx_test = torch.LongTensor(self.idx_test)
        if config.learning.method_learning == "GCN":
            self.model = GCN(nfeat=self.features.shape[1],
                             nhid=config.learning.hidden,
                             nclass=self.labels.max().item(),
                             dropout=config.learning.dropout)

            if config.learning.cuda:
                self.adj = self.adj.cuda()

        elif config.learning.method_learning == "AGNN":
            self.model = AGNN(nfeat=self.features.shape[1],
                              nhid=config.learning.hidden,
                              nclass=self.labels.max().item(),
                              nlayers=config.learning.layers,
                              dropout_rate=config.learning.dropout)

        if config.learning.cuda:
            self.model.cuda()
            self.features = self.features.cuda()
            # self.adj = self.adj.cuda()
            self.all_labels = self.all_labels.cuda()
            self.idx_train = self.idx_train.cuda()
            self.idx_test = self.idx_test.cuda()
            self.idx_val = self.idx_val.cuda()

    def train(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=config.learning.lr,
                               weight_decay=config.learning.weight_decay)
        max_acc = 0
        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            output = self.model(self.features, self.adj)
            loss_train = F.nll_loss(output[self.idx_train], self.all_labels[self.idx_train])
            # acc_train = accuracy(output[self.idx_train], self.all_labels[self.idx_train])
            loss_train.backward()
            optimizer.step()
            self.model.eval()
            output = self.model(self.features, self.adj)
            #acc_val = accuracy(output[self.idx_val], self.all_labels_init[self.idx_val])
            #if acc_val.item() >= max_acc:
            #    max_acc = acc_val.item()
            acc_test = accuracy(output[self.idx_test], self.all_labels_init[self.idx_test])
            if acc_test.item() > max_acc:
                max_acc = acc_test.item()
                if config.learning.save_model:
                    torch.save(self.model.state_dict(),
                               config.paths.models)
                self.best_epoch = epoch
                acc_test = accuracy(output[self.idx_test], self.all_labels_init[self.idx_test])
                preds = output
                beliefs = preds.max(1)[1].type_as(self.all_labels)
                #torch.save(self.model.state_dict(),
                #               "../Stats/models_graph/loss/model{}_methodmix_{}_ration_{}_unkn.h5".format(
                #                   config.method_decomposition_embedding, val, config.num_nearest_neighbours))
        return beliefs.numpy() +1 , acc_test
