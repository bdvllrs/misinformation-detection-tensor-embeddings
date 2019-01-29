from utils import embedding_matrix_2_kNN, get_rate, precision, recall, f1_score, accuracy2
from utils.ArticlesHandler import ArticlesHandler
from utils import Config
import time
import numpy as np
from pygcn.utils import accuracy

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


for meth in enumerate(methods):
    debut_meth = time.time()
    config.set("embedding.method_decomposition_embedding", meth)
    handler = ArticlesHandler(config)
    for meth2 in enumerate(methods):
        print("Methods : ", str(meth2))
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
        # select_labels = SelectLabelsPostprocessor(config, handler.articles)
        # handler.add_postprocessing(select_labels, "label-selection")
        # handler.postprocess()
        labels = handler.articles.labels
        all_labels = handler.articles.labels_untouched

        if meth == meth2:
            C_nodes = C.copy()
        else:
            config.set("method_decomposition_embedding", meth2)
            C_nodes = handler.get_tensor()

        C, C_nodes, labels, all_labels = list(
            zip(*np.random.permutation(list(zip(C, C_nodes, labels, all_labels)))))



        for i, val in enumerate(pourcentage_know):
            print("Pourcentage : ", str(val))
            num_unknown_labels = nbre_total_article - int(val / 100 * nbre_total_article)
            acc2 = []
            prec2 = []
            rec2 = []
            f12 = []
            times2 = []
            best_epochs2 = []
            for acc_repeat in range(config.stats.iteration_stat):
                acc = []
                prec = []
                rec = []
                f1 = []
                times= []
                best_epochs = []
                all_labels_init = all_labels
                labels_init = list(all_labels)
                for k_num in range(num_unknown_labels):
                    labels_init[k_num] = 0
                C, C_nodes, labels_init, all_labels_init = list(
                    zip(*np.random.permutation(list(zip(C, C_nodes, labels_init, all_labels_init)))))
                for j, val2 in enumerate(pourcentage_voisin):
                    num_nearest_neighbours = int(val2)
                    assert nbre_total_article >= num_nearest_neighbours, "Can't have more neighbours than nodes!"
                    graph = embedding_matrix_2_kNN(C, k=config.graph.num_nearest_neighbours).toarray()
                    trainer = TrainerGraph(C_nodes, graph, all_labels, labels)
                    beliefs = trainer.train()
                    # Compute hit rate
                    beliefs[beliefs > 0] = 1
                    beliefs[beliefs < 0] = -1
                    TP, TN, FP, FN = get_rate(beliefs, labels, all_labels)
                    acc = accuracy2(TP, TN, FP, FN)
                    prec = precision(TP, FP)
                    rec = recall(TP, FN)
                    f1 = f1_score(prec, rec)
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
        np.save('../Stats/{}_{}_methodmix_{}_ration_f1_score_val stats_mean'.format(meth, meth2, ratio),
                best_epoch_score_mean)
        np.save('../Stats/{}_{}_methodmix_{}_ration_best_epoch_score_val stats_std'.format(meth, meth2, ratio),
                best_epoch_score_std)
        np.save('../Stats/{}_{}_methodmix_{}_ration_time_score_val stats_mean'.format(meth, meth2, ratio),
                times_score_mean)
        np.save('../Stats/{}_{}_methodmix_{}_ration_time_score_val stats_std'.format(meth, meth2, ratio),
                times_score_std)

        print(time.time() - debut)
    print('temps method : ', meth)
    print(time.time() - debut_meth)