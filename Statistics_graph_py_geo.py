from utils import embedding_matrix_2_kNN, get_rate, precision, recall, f1_score, accuracy2
from utils.ArticleTensor import ArticleTensor
from utils.ArticlesHandler import ArticlesHandler
from utils import Config
import time
import numpy as np
from pygcn.utils import accuracy, load_from_features, encode_onehot, normalize, sparse_mx_to_torch_sparse_tensor
from torch_geometric.utils import sparse as S
import torch
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.nn import AGNNConv

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = torch.nn.Linear(num_features, 16)

        self.prop1 = AGNNConv(requires_grad=False)
        self.prop2 = AGNNConv(requires_grad=True)
        self.lin2 = torch.nn.Linear(16, num_classes)

        self.prop1_mat_weight = torch.tensor(0)
        self.prop2_mat_weight = torch.tensor(0)

        self.edge_index_with_self_lopp = torch.tensor(0)

    def forward(self):
        x = F.dropout(data.x, 0.5, training=self.training)
        x = F.relu(self.lin1(data.x))

        self.edge_index_with_self_loop, self.prop1_mat_weight = self.prop1.propagation_matrix(x, data.edge_index)
        x = self.prop1(x, data.edge_index)

        _, self.prop2_mat_weight = self.prop2.propagation_matrix(x, data.edge_index)
        x = self.prop2(x, data.edge_index)

        x = F.dropout(x, 0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


seed = 12

np.random.seed(seed=seed)

layers_test = [2,3,4]
cuda = False
hidden = 16
dropout = 0.5
lr = 0.01
weight_decay = 5e-4
fastmode = False
epochs = 1000#450


config = Config(file='config')
pourcentage_know = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 95]


#"""
articleTensor = ArticleTensor(config.config)
articleTensor.get_articles(config["dataset_name"], number_fake=config['num_fake_articles'],
                           number_real=config['num_real_articles'])
articleTensor.build_word_to_index(max_words=config['vocab_size'])

nbre_total_article = config['num_real_articles'] + config['num_fake_articles']
pourcentage_voisin = np.array([1, 2, 3, 4, 5, 6 ,7, 8, 9])
ratios = [0.75]
methods = [("GloVe", "mean"), ("decomposition", False)]


for meth in enumerate(methods):
    debut_meth = time.time()
    for layers in layers_test:
        debut = time.time()
        ratio = 0.75
        accuracy_mean = np.zeros((len(pourcentage_know), len(pourcentage_voisin)))
        accuracy_std = np.zeros((len(pourcentage_know), len(pourcentage_voisin)))
        precision_mean = np.zeros((len(pourcentage_know), len(pourcentage_voisin)))
        precision_std = np.zeros((len(pourcentage_know), len(pourcentage_voisin)))
        recall_mean = np.zeros((len(pourcentage_know), len(pourcentage_voisin)))
        recall_std = np.zeros((len(pourcentage_know), len(pourcentage_voisin)))
        f1_score_mean = np.zeros((len(pourcentage_know), len(pourcentage_voisin)))
        f1_score_std = np.zeros((len(pourcentage_know), len(pourcentage_voisin)))
        if meth[1][0] == "decomposition":
            tensor, labels, all_labels_init = articleTensor.get_tensor_coocurrence(
                window=config['size_word_co_occurrence_window'],
                num_unknown=0,
                ratio=ratio,
                use_frequency=meth[1][1])
            _, (_, _, C) = ArticleTensor.get_parafac_decomposition(tensor, rank=config['rank_parafac_decomposition'])
        if meth[1][0] == "GloVe":
            tensor, labels_init, all_labels_init = articleTensor.get_tensor_Glove(meth[1][1],
                                                                        ratio,
                                                                        num_unknown=0)
            C = np.transpose(tensor)
        print(meth, layers)
        for i, val in enumerate(pourcentage_know):
            num_unknown_labels = nbre_total_article - int(val / 100 * nbre_total_article)
            acc2 = []
            prec2 = []
            rec2 = []
            f12 = []
            for acc_repeat in range(config["iteration_stat"]):
                acc = []
                prec = []
                rec = []
                f1 = []
                labels_init = list(all_labels_init)
                for k_num in range(num_unknown_labels):
                    labels_init[k_num] = 0
                C, labels_init, all_labels_init = list(
                    zip(*np.random.permutation(list(zip(C, labels_init, all_labels_init)))))
                for j, val2 in enumerate(pourcentage_voisin):
                    num_nearest_neighbours = int(val2)
                    assert nbre_total_article >= num_nearest_neighbours, "Can't have more neighbours than nodes!"
                    graph = embedding_matrix_2_kNN(C, k=config.num_nearest_neighbours).toarray()
                    adj = sp.coo_matrix(graph, dtype=np.float32)
                    all_labels = encode_onehot(all_labels_init)
                    features = normalize(np.array(C))
                    adj = normalize(adj + sp.eye(adj.shape[0]))
                    features = torch.FloatTensor(np.array(features))
                    all_labels = torch.LongTensor(np.where(all_labels)[1])
                    adj = sparse_mx_to_torch_sparse_tensor(adj)
                    idx_test = np.where(np.array(labels_init)==0)[0]
                    labels = encode_onehot(labels_init)
                    labels = torch.LongTensor(np.where(labels)[1])
                    idx_train = np.where(labels)[0]
                    idx_train = torch.LongTensor(idx_train)
                    idx_test = torch.LongTensor(idx_test)

                    num_features = features.shape[1]
                    num_classes = 2
                    adj_dense = adj.to_dense()
                    index, value = S.dense_to_sparse(adj_dense)


                    data = Data(x=features, edge_index=index)

                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    # device = 'cpu'
                    model = Net().to(device)



                    optimizer = optim.Adam(model.parameters(),
                                           lr=lr, weight_decay=weight_decay)

                    if cuda:
                        model.cuda()
                        features = features.cuda()
                        adj = adj.cuda()
                        all_labels = all_labels.cuda()
                        idx_train = idx_train.cuda()
                        idx_test = idx_test.cuda()

                    t_total = time.time()
                    loss_min = 100
                    max_acc = 0
                    for epoch in range(epochs):
                        model.train()
                        optimizer.zero_grad()
                        output = model(features, adj)
                        loss_train = F.nll_loss(output[idx_train], all_labels[idx_train])
                        #acc_train = accuracy(output[idx_train], all_labels[idx_train])
                        loss_train.backward()
                        optimizer.step()
                        model.eval()
                        output = model(features, adj)
                        acc_test = accuracy(output[idx_test], all_labels[idx_test])
                        if acc_test.item() > max_acc:
                            max_acc = acc_test.item()
                            torch.save(model.state_dict(),
                                       "../Stats/models_graph/acc/model{}_method_{}_ration_{}_unkn_{}_layers_{}_acc.h5".format(
                                           config.method_decomposition_embedding, val, config.num_nearest_neighbours, layers, max_acc))
                            best_epoch = epoch
                            beliefs = output.max(1)[1].type_as(labels).numpy()
                            beliefs[beliefs == 1] = -1
                            beliefs[beliefs == 0] = 1
                            TP, TN, FP, FN = get_rate(beliefs, labels_init, all_labels_init)
                            if len(acc)!=j:
                                acc.pop()
                                prec.pop()
                                rec.pop()
                                f1.pop()
                            acc.append(accuracy2(TP, TN, FP, FN))
                            prec.append(precision(TP, FP))
                            rec.append(recall(TP, FN))
                            f1.append(f1_score(prec[-1], rec[-1]))
                        #if loss_min > loss_train.item():
                        #    torch.save(model.state_dict(),
                        #               "../Stats/models_graph/loss/model{}_method_{}_ration_{}_unkn.h5".format(
                        #                   config.method_decomposition_embedding, val, config.num_nearest_neighbours))
                        #    loss_min = loss_train.item()
                    print("End training, in ", time.time()-t_total)
                    print("Best epoch : ", best_epoch, max_acc)
                acc2.append(acc)
                prec2.append(prec)
                rec2.append(rec)
                f12.append(f1)
            accuracy_mean[i, :] = np.array(acc2).mean(axis=0)
            accuracy_std[i, :] = np.array(acc2).std(axis=0)
            precision_mean[i, :] = np.array(prec2).mean(axis=0)
            precision_std[i, :] = np.array(prec2).std(axis=0)
            recall_mean[i, :] = np.array(rec2).mean(axis=0)
            recall_std[i, :] = np.array(rec2).std(axis=0)
            f1_score_mean[i, :] = np.array(f12).mean(axis=0)
            f1_score_std[i, :] = np.array(f12).std(axis=0)
        print('save_model')
        np.save('../Stats/{}_{}_method_{}_ration_accuracy_val stats_mean'.format(meth[1][0], meth[1][1], layers),
                accuracy_mean)
        np.save('../Stats/{}_{}_method_{}_ration_accuracy_val stats_std'.format(meth[1][0], meth[1][1], layers),
                accuracy_std)
        np.save('../Stats/{}_{}_method_{}_ration_precision_val stats_mean'.format(meth[1][0], meth[1][1], layers),
                precision_mean)
        np.save('../Stats/{}_{}_method_{}_ration_precision_val stats_std'.format(meth[1][0], meth[1][1], layers),
                precision_std)
        np.save('../Stats/{}_{}_method_{}_ration_recall_val stats_mean'.format(meth[1][0], meth[1][1], layers), recall_mean)
        np.save('../Stats/{}_{}_method_{}_ration_recall_val stats_std'.format(meth[1][0], meth[1][1], layers), recall_std)
        np.save('../Stats/{}_{}_method_{}_ration_f1_score_val stats_mean'.format(meth[1][0], meth[1][1], layers),
                f1_score_mean)
        np.save('../Stats/{}_{}_method_{}_ration_f1_score_val stats_std'.format(meth[1][0], meth[1][1], layers),
                f1_score_std)
        print(time.time() - debut)
    print('temps method : ', meth)
    print(time.time() - debut_meth)

