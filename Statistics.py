from ArticleTensor import ArticleTensor
from utils import solve, embedding_matrix_2_kNN, load_config, get_rate, accuracy, precision, recall, f1_score
import time
import numpy as np

config = load_config()
articleTensor = ArticleTensor(config['dataset_path'])
articleTensor.get_articles(config['dataset_name'], number_fake=config['num_fake_articles'],
                           number_real=config['num_real_articles'])
articleTensor.build_word_to_index(max_words=config['vocab_size'])

nbre_total_article= config['num_real_articles']+config['num_fake_articles']
pourcentage_know=[2,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95]
pourcentage_voisin=np.linspace(1,100,10)
ratios=[0.5,0.55,0.6,0.65,0.7,0.75,0.8]
methods=[("decomposition",False),("decomposition",True),("GloVe","mean"),("GloVe","RNN")]

for meth in enumerate(methods):
    debut_meth=time.time()
    for k in ratios:
        debut = time.time()
        ratio = k
        accuracy_mean=np.zeros((len(pourcentage_know),len(pourcentage_voisin)))
        accuracy_std=np.zeros((len(pourcentage_know),len(pourcentage_voisin)))
        precision_mean=np.zeros((len(pourcentage_know),len(pourcentage_voisin)))
        precision_std=np.zeros((len(pourcentage_know),len(pourcentage_voisin)))
        recall_mean=np.zeros((len(pourcentage_know),len(pourcentage_voisin)))
        recall_std=np.zeros((len(pourcentage_know),len(pourcentage_voisin)))
        f1_score_mean=np.zeros((len(pourcentage_know),len(pourcentage_voisin)))
        f1_score_std=np.zeros((len(pourcentage_know),len(pourcentage_voisin)))
        print(meth, k)
        for i, val in enumerate(pourcentage_know):
            num_unknown_labels=nbre_total_article-int(val/100*nbre_total_article)
            for j, val2 in enumerate(pourcentage_voisin):
                acc=[]
                prec=[]
                rec=[]
                f1=[]
                num_nearest_neighbours=int(val2)
                assert nbre_total_article >= num_nearest_neighbours, "Can't have more neighbours than nodes!"
                for acc_repeat in range(config["iteration_stat"]):
                    if meth[1][0]=="decomposition":
                        tensor, labels, all_labels = articleTensor.get_tensor_coocurrence(window=config['size_word_co_occurrence_window'],
                                                                                          num_unknown=num_unknown_labels,
                                                                                          ratio=ratio,
                                                                                          use_frequency=meth[1][1])
                        _, (_, _, C) = ArticleTensor.get_parafac_decomposition(tensor, rank=config['rank_parafac_decomposition'])
                    if meth[1][0]=="GloVe":
                        tensor, labels, all_labels = articleTensor.get_tensor_Glove(meth[1][1],
                                                                                    config["vocab_util_pourcentage"], ratio=ratio,
                                                                                    num_unknown=num_unknown_labels)
                        C = np.transpose(tensor)

                    graph = embedding_matrix_2_kNN(C, k=num_nearest_neighbours).toarray()
                    beliefs = solve(graph, labels)
                    beliefs[beliefs > 0] = 1
                    beliefs[beliefs < 0] = -1
                    TP, TN, FP, FN=get_rate(beliefs,labels, all_labels)
                    acc.append(accuracy(TP, TN, FP, FN))
                    prec.append(precision(TP, FP))
                    rec.append(recall(TP, FN))
                    f1.append(f1_score( prec[-1],rec[-1]))
                accuracy_mean[i, j] = np.mean(acc)
                accuracy_std[i, j] = np.std(acc)
                precision_mean[i, j] = np.mean(prec)
                precision_std[i, j] = np.std(prec)
                recall_mean[i, j] = np.mean(rec)
                recall_std[i, j] = np.std(rec)
                f1_score_mean[i, j] = np.mean(f1)
                f1_score_std[i, j] = np.std(f1)
            print('save_model')
            np.save('../Stats/{}_method_{}_ration_accuracy_val stats_mean'.format(meth[1],k), accuracy_mean)
            np.save('../Stats/{}_method_{}_ration_accuracy_val stats_std'.format(meth[1],k), accuracy_std)
            np.save('../Stats/{}_method_{}_ration_precision_val stats_mean'.format(meth[1],k), accuracy_mean)
            np.save('../Stats/{}_method_{}_ration_precision_val stats_std'.format(meth[1],k), accuracy_std)
            np.save('../Stats/{}_method_{}_ration_recall_val stats_mean'.format(meth[1],k), accuracy_mean)
            np.save('../Stats/{}_method_{}_ration_recall_val stats_std'.format(meth[1],k), accuracy_std)
            np.save('../Stats/{}_method_{}_ration_f1_score_val stats_mean'.format(meth[1],k), accuracy_mean)
            np.save('../Stats/{}_method_{}_ration_f1_score_val stats_std'.format(meth[1],k), accuracy_std)
        print(time.time() - debut)
    print('temps method : ', meth)
    print(time.time()-debut_meth)