from utils.ArticleTensor import ArticleTensor
from utils import solve, embedding_matrix_2_kNN, get_rate, accuracy, precision, recall, f1_score
from utils.Config import Config
import time
import numpy as np

config = Config(file='config')
articleTensor = ArticleTensor(config.config)
articleTensor.get_articles(config["dataset_name"], number_fake=config['num_fake_articles'],
                           number_real=config['num_real_articles'])
articleTensor.build_word_to_index(max_words=config['vocab_size'])

nbre_total_article = config['num_real_articles'] + config['num_fake_articles']
pourcentage_know = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
pourcentage_voisin = np.array([1, 2, 3, 4, 5, 6 ,7, 8, 9, 10, 15])
ratios = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
methods = [("decomposition", False),  ("GloVe", "mean"), (),()]

for meth in enumerate(methods):
    debut_meth = time.time()
    for k in ratios:
        debut = time.time()
        ratio = k
        accuracy_mean = np.zeros((len(pourcentage_know), len(pourcentage_voisin)))
        accuracy_std = np.zeros((len(pourcentage_know), len(pourcentage_voisin)))
        precision_mean = np.zeros((len(pourcentage_know), len(pourcentage_voisin)))
        precision_std = np.zeros((len(pourcentage_know), len(pourcentage_voisin)))
        recall_mean = np.zeros((len(pourcentage_know), len(pourcentage_voisin)))
        recall_std = np.zeros((len(pourcentage_know), len(pourcentage_voisin)))
        f1_score_mean = np.zeros((len(pourcentage_know), len(pourcentage_voisin)))
        f1_score_std = np.zeros((len(pourcentage_know), len(pourcentage_voisin)))
        if meth[1][0] == "decomposition":
            tensor, labels, all_labels = articleTensor.get_tensor_coocurrence(
                window=config['size_word_co_occurrence_window'],
                num_unknown=0,
                ratio=ratio,
                use_frequency=meth[1][1])
            _, (_, _, C) = ArticleTensor.get_parafac_decomposition(tensor, rank=config['rank_parafac_decomposition'])
        if meth[1][0] == "GloVe":
            tensor, labels, all_labels = articleTensor.get_tensor_Glove(meth[1][1],
                                                                        ratio,
                                                                        num_unknown=0)
            C = np.transpose(tensor)
        print(meth, k)
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
                labels = list(all_labels)
                for k_num in range(num_unknown_labels):
                    labels[k_num] = 0
                C, labels, all_labels = list(
                    zip(*np.random.permutation(list(zip(C, labels, all_labels)))))
                for j, val2 in enumerate(pourcentage_voisin):
                    num_nearest_neighbours = int(val2)
                    assert nbre_total_article >= num_nearest_neighbours, "Can't have more neighbours than nodes!"

                    graph = embedding_matrix_2_kNN(C, k=num_nearest_neighbours).toarray()
                    beliefs = solve(graph, labels)
                    beliefs[beliefs > 0] = 1
                    beliefs[beliefs < 0] = -1
                    TP, TN, FP, FN = get_rate(beliefs, labels, all_labels)
                    acc.append(accuracy(TP, TN, FP, FN))
                    prec.append(precision(TP, FP))
                    rec.append(recall(TP, FN))
                    f1.append(f1_score(prec[-1], rec[-1]))
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
        np.save('../Stats/{}_{}_method_{}_ration_accuracy_val stats_mean'.format(meth[1][0], meth[1][1], k),
                accuracy_mean)
        np.save('../Stats/{}_{}_method_{}_ration_accuracy_val stats_std'.format(meth[1][0], meth[1][1], k),
                accuracy_std)
        np.save('../Stats/{}_{}_method_{}_ration_precision_val stats_mean'.format(meth[1][0], meth[1][1], k),
                precision_mean)
        np.save('../Stats/{}_{}_method_{}_ration_precision_val stats_std'.format(meth[1][0], meth[1][1], k),
                precision_std)
        np.save('../Stats/{}_{}_method_{}_ration_recall_val stats_mean'.format(meth[1][0], meth[1][1], k), recall_mean)
        np.save('../Stats/{}_{}_method_{}_ration_recall_val stats_std'.format(meth[1][0], meth[1][1], k), recall_std)
        np.save('../Stats/{}_{}_method_{}_ration_f1_score_val stats_mean'.format(meth[1][0], meth[1][1], k),
                f1_score_mean)
        np.save('../Stats/{}_{}_method_{}_ration_f1_score_val stats_std'.format(meth[1][0], meth[1][1], k),
                f1_score_std)
        print(time.time() - debut)
    print('temps method : ', meth)
    print(time.time() - debut_meth)
