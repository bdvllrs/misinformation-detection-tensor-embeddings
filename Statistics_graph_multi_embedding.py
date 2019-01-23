from utils import embedding_matrix_2_kNN, get_rate, precision, recall, f1_score, accuracy2
from utils.ArticleTensor import ArticleTensor
from utils.ArticlesHandler import ArticlesHandler
from utils import Config
import time
import numpy as np
from pygcn.utils import accuracy, load_from_features, encode_onehot, normalize, sparse_mx_to_torch_sparse_tensor
from pyagnn.agnn.model import AGNN
import torch
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp
import torch

seed = 12

np.random.seed(seed=seed)

layers_test = [2]
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
pourcentage_voisin = np.array([2,3,4,5])
ratios = [0.75]
methods = [("Glove", "mean")]


for meth in enumerate(methods):
    debut_meth = time.time()
    for layers in layers_test:
        print("Layers : ", str(layers))
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
        best_epoch_score_mean = np.zeros((len(pourcentage_know), len(pourcentage_voisin)))
        best_epoch_score_std = np.zeros((len(pourcentage_know), len(pourcentage_voisin)))
        times_score_mean = np.zeros((len(pourcentage_know), len(pourcentage_voisin)))
        times_score_std = np.zeros((len(pourcentage_know), len(pourcentage_voisin)))

        tensor, labels, all_labels_init = articleTensor.get_tensor_coocurrence(
            window=config['size_word_co_occurrence_window'],
            num_unknown=0,
            ratio=ratio,
            use_frequency=meth[1][1])
        _, (_, _, C) = ArticleTensor.get_parafac_decomposition(tensor, rank=config['rank_parafac_decomposition'])


        tensor, labels_init, all_labels_init = articleTensor.get_tensor_Glove(meth[1][1],
                                                                    ratio,
                                                                    num_unknown=0)
        C_node = np.transpose(tensor)
        """
        if meth[1][0] == "LDA":
            config.set("method_decomposition_embedding", "LDA")
            print("Loading dataset", config.dataset_name)
            articles = ArticlesHandler(config)
            print("Performing decomposition...")
            C = articles.get_tensor()

            print(C.shape)

            labels_init = articles.articles.labels
            all_labels_init = articles.articles.labels_untouched

        if meth[1][0] == "Transformer":
            config.set("method_decomposition_embedding", "Transformer")
            print("Loading dataset", config.dataset_name)
            articles = ArticlesHandler(config)
            print("Performing decomposition...")
            C = articles.get_tensor().detach().numpy()
            labels_init = articles.articles.labels
            all_labels_init = articles.articles.labels_untouched
        """
        print(meth, layers)
        for i, val in enumerate(pourcentage_know):
            print("Pourcentage : ", str(val))
            num_unknown_labels = nbre_total_article - int(val / 100 * nbre_total_article)
            acc2 = []
            prec2 = []
            rec2 = []
            f12 = []
            times2 = []
            best_epochs2 = []
            for acc_repeat in range(config["iteration_stat"]):
                acc = []
                prec = []
                rec = []
                f1 = []
                times= []
                best_epochs = []
                labels_init = list(all_labels_init)
                for k_num in range(num_unknown_labels):
                    labels_init[k_num] = 0
                C, C_node, labels_init, all_labels_init = list(
                    zip(*np.random.permutation(list(zip(C, C_node, labels_init, all_labels_init)))))
                for j, val2 in enumerate(pourcentage_voisin):
                    num_nearest_neighbours = int(val2)
                    assert nbre_total_article >= num_nearest_neighbours, "Can't have more neighbours than nodes!"
                    graph = embedding_matrix_2_kNN(C, k=config.num_nearest_neighbours).toarray()
                    adj = sp.coo_matrix(graph, dtype=np.float32)
                    all_labels = encode_onehot(all_labels_init)
                    features = normalize(np.array(C_node))
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
                    model = AGNN(nfeat=features.shape[1],
                                 nhid=hidden,
                                 nclass=2,
                                 nlayers=layers,
                                 dropout_rate=0.5)

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
                                       "../Stats/models_graph/acc/model{}_methodmix_{}_ration_{}_unkn_{}_layers_{}_acc.h5".format(
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
                                best_epochs.pop()
                                times.pop()
                            acc.append(accuracy2(TP, TN, FP, FN))
                            prec.append(precision(TP, FP))
                            rec.append(recall(TP, FN))
                            f1.append(f1_score(prec[-1], rec[-1]))
                            best_epochs.append(best_epoch)
                            times.append(time.time()-t_total)
                        #if loss_min > loss_train.item():
                        #    torch.save(model.state_dict(),
                        #               "../Stats/models_graph/loss/model{}_methodmix_{}_ration_{}_unkn.h5".format(
                        #                   config.method_decomposition_embedding, val, config.num_nearest_neighbours))
                        #    loss_min = loss_train.item()
                    #print("End training, in ", time.time()-t_total)
                    #print("Best epoch : ", best_epoch, max_acc)
                acc2.append(acc)
                prec2.append(prec)
                rec2.append(rec)
                f12.append(f1)
                best_epochs2.append(best_epochs)
                times2.append(times)

            accuracy_mean[i, :] = np.array(acc2).mean(axis=0)
            accuracy_std[i, :] = np.array(acc2).std(axis=0)
            precision_mean[i, :] = np.array(prec2).mean(axis=0)
            precision_std[i, :] = np.array(prec2).std(axis=0)
            recall_mean[i, :] = np.array(rec2).mean(axis=0)
            recall_std[i, :] = np.array(rec2).std(axis=0)
            f1_score_mean[i, :] = np.array(f12).mean(axis=0)
            f1_score_std[i, :] = np.array(f12).std(axis=0)
            best_epoch_score_mean[i, :] = np.array(best_epochs2).mean(axis=0)
            best_epoch_score_std[i, :] = np.array(best_epochs2).std(axis=0)
            times_score_mean[i, :] = np.array(times2).mean(axis=0)
            times_score_std[i, :] = np.array(times2).std(axis=0)


        print('save_model')
        np.save('../Stats/{}_{}_methodmix_{}_ration_accuracy_val stats_mean'.format(meth[1][0], meth[1][1], layers),
                accuracy_mean)
        np.save('../Stats/{}_{}_methodmix_{}_ration_accuracy_val stats_std'.format(meth[1][0], meth[1][1], layers),
                accuracy_std)
        np.save('../Stats/{}_{}_methodmix_{}_ration_precision_val stats_mean'.format(meth[1][0], meth[1][1], layers),
                precision_mean)
        np.save('../Stats/{}_{}_methodmix_{}_ration_precision_val stats_std'.format(meth[1][0], meth[1][1], layers),
                precision_std)
        np.save('../Stats/{}_{}_methodmix_{}_ration_recall_val stats_mean'.format(meth[1][0], meth[1][1], layers), recall_mean)
        np.save('../Stats/{}_{}_methodmix_{}_ration_recall_val stats_std'.format(meth[1][0], meth[1][1], layers), recall_std)
        np.save('../Stats/{}_{}_methodmix_{}_ration_f1_score_val stats_mean'.format(meth[1][0], meth[1][1], layers),
                f1_score_mean)
        np.save('../Stats/{}_{}_methodmix_{}_ration_f1_score_val stats_std'.format(meth[1][0], meth[1][1], layers),
                f1_score_std)
        np.save('../Stats/{}_{}_methodmix_{}_ration_f1_score_val stats_mean'.format(meth[1][0], meth[1][1], layers),
                best_epoch_score_mean)
        np.save('../Stats/{}_{}_methodmix_{}_ration_best_epoch_score_val stats_std'.format(meth[1][0], meth[1][1], layers),
                best_epoch_score_std)
        np.save('../Stats/{}_{}_methodmix_{}_ration_time_score_val stats_mean'.format(meth[1][0], meth[1][1], layers),
                times_score_mean)
        np.save('../Stats/{}_{}_methodmix_{}_ration_time_score_val stats_std'.format(meth[1][0], meth[1][1], layers),
                times_score_std)

        print(time.time() - debut)
    print('temps method : ', meth)
    print(time.time() - debut_meth)

