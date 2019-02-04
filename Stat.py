from utils import embedding_matrix_2_kNN, get_rate, precision, recall, f1_score, accuracy2, solve
from utils.ArticlesHandler import ArticlesHandler
from utils import Config
import time
import numpy as np
from utils.Trainer_graph import TrainerGraph

config = Config('config/')
seed = 12
np.random.seed(seed=seed)
cuda = config.learning.cuda
hidden = config.learning.hidden
dropout = config.learning.dropout
lr = config.learning.lr
weight_decay = config.learning.weight_decay
fastmode = config.learning.fastmode
epochs = config.learning.epochs
pourcentage_know = config.stats.pourcentage_know
pourcentage_voisin = config.stats.pourcentage_voisin
layers_test = config.stats.layers_test
ratio = config.embedding.vocab_util_pourcentage
handler = ArticlesHandler(config)
nbre_total_article = config.stats.num_real_articles + config.stats.num_fake_articles
layers = config.learning.layers
methods = config.stats.methods_1



for idx_meth, meth in enumerate(methods):
    debut_meth = time.time()
    config.embedding.set("method_decomposition_embedding", meth)
    handler = ArticlesHandler(config)
    for idx_meth2, meth2 in enumerate(methods):
        print("Methods : ", str(meth2))
        print(meth)
        debut = time.time()
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
        C = handler.get_tensor()
        all_labels = handler.articles.labels_untouched
        if meth == meth2:
            C_nodes = C.copy()
        else:
            config.embedding.set("method_decomposition_embedding", meth2)
            C_nodes = handler.get_tensor()
        for i, val in enumerate(pourcentage_know):
            print("Pourcentage : ", str(val))
            num_unknown_labels = nbre_total_article - int(val / 100 * nbre_total_article)
            acc2 = []
            prec2 = []
            rec2 = []
            f12 = []
            times2 = []
            best_epochs2 = []
            for j, val2 in enumerate(pourcentage_voisin):
                num_nearest_neighbours = int(val2)
                assert nbre_total_article >= num_nearest_neighbours, "Can't have more neighbours than nodes!"
                graph = embedding_matrix_2_kNN(C, k=num_nearest_neighbours).toarray()
                acc = []
                prec = []
                rec = []
                f1 = []
                times= []
                best_epochs = []
                for acc_repeat in range(config.stats.iteration_stat):
                    labels = np.array(all_labels).copy()
                    val_labels = np.unique(labels)
                    zero_idx_1 = []
                    for idx, lab in enumerate(val_labels):
                        n1 = np.where(labels == lab)[0]
                        random_idx_1 = np.random.permutation(n1)
                        zero_idx_1.append(random_idx_1[:int(num_unknown_labels/len(val_labels))])
                    zero_idx = np.concatenate(zero_idx_1)
                    labels[zero_idx] = 0
                    if config.learning.method_learning == "FaBP":
                        # classe  b(i){> 0, < 0} means i ∈ {“+”, “-”}
                        l = np.array(labels)
                        beliefs = solve(graph, l)
                        fin4 = time.time()
                    else:
                        trainer = TrainerGraph(C_nodes, graph, all_labels, labels)
                        beliefs, acc_test = trainer.train()
                        #print(acc_test)
                        fin4 = time.time()
                        # Compute hit rate
                        # TODO: changer pour le multiclasse...
                        #beliefs[beliefs > 0] = 1
                        #beliefs[beliefs < 0] = 2
                    TP, TN, FP, FN = get_rate(beliefs, labels, all_labels)
                    acc.append(accuracy2(TP, TN, FP, FN))
                    prec.append(precision(TP, FP))
                    rec.append(recall(TP, FN))
                    f1.append(f1_score(prec[-1], rec[-1]))
                acc2.append(acc)
                prec2.append(prec)
                rec2.append(rec)
                f12.append(f1)
                best_epochs2.append(best_epochs)
                times2.append(times)
            accuracy_mean[i, :] = np.array(acc2).mean(axis=1)
            accuracy_std[i, :] = np.array(acc2).std(axis=1)
            precision_mean[i, :] = np.array(prec2).mean(axis=1)
            precision_std[i, :] = np.array(prec2).std(axis=1)
            recall_mean[i, :] = np.array(rec2).mean(axis=1)
            recall_std[i, :] = np.array(rec2).std(axis=1)
            f1_score_mean[i, :] = np.array(f12).mean(axis=1)
            f1_score_std[i, :] = np.array(f12).std(axis=1)

            print(accuracy_mean)
            print(accuracy_std)

        print('save_model')
        np.save('../Stats/{}_{}_methodmix_{}_ration_accuracy_val stats_mean'.format(meth, meth2, ratio),
                accuracy_mean)
        np.save('../Stats/{}_{}_methodmix_{}_ration_accuracy_val stats_std'.format(meth, meth2, ratio),
                accuracy_std)
        np.save('../Stats/{}_{}_methodmix_{}_ration_precision_val stats_mean'.format(meth, meth2, ratio),
                precision_mean)
        np.save('../Stats/{}_{}_methodmix_{}_ration_precision_val stats_std'.format(meth, meth2, ratio),
                precision_std)
        np.save('../Stats/{}_{}_methodmix_{}_ration_recall_val stats_mean'.format(meth, meth2, ratio), recall_mean)
        np.save('../Stats/{}_{}_methodmix_{}_ration_recall_val stats_std'.format(meth, meth2, ratio), recall_std)
        np.save('../Stats/{}_{}_methodmix_{}_ration_f1_score_val stats_mean'.format(meth, meth2, ratio),
                f1_score_mean)
        np.save('../Stats/{}_{}_methodmix_{}_ration_f1_score_val stats_std'.format(meth, meth2, ratio),
                f1_score_std)
        print(time.time() - debut)
    print('temps method : ', meth)
    print(time.time() - debut_meth)