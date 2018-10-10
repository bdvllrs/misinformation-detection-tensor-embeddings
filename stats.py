from ArticleTensor import ArticleTensor
from kNN import embedding_matrix_2_kNN
from utils import solve
import time
import numpy as np

num_nearest_neighbours = 2
num_fake_articles = 128
num_real_articles = 128
num_unknown_labels = 40
vocab_size = -1
method_embedding_glove="mean"

assert num_fake_articles + num_real_articles > num_nearest_neighbours, "Can't have more neighbours than nodes!"

debut = time.time()
articleTensor = ArticleTensor('../Dataset/fakenewsdata1/Public Data')
articleTensor.get_articles('Buzzfeed Political News Dataset', number_fake=num_fake_articles,
                           number_real=num_real_articles)
articleTensor.build_word_to_index(max_words=vocab_size)

pourcentage_know=[2,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95]
pourcentage_voisin=np.linspace(1,100,10)
ratios=[0.5,0.55,0.6,0.65,0.7,0.75,0.8]

for k in ratios:
    ratio = k
    accuracy_mean=np.zeros((len(pourcentage_know),len(pourcentage_voisin)))
    accuracy_std=np.zeros((len(pourcentage_know),len(pourcentage_voisin)))
    print(k)
    for i, val in enumerate(pourcentage_know):
        num_unknown_labels=256-int(val/100*256)
        for j, val2 in enumerate(pourcentage_voisin):
            acc=[]
            num_nearest_neighbours=int(val2)
            for acc_repeat in range(20):
                tensor, labels, all_labels = articleTensor.get_tensor(method_embedding_glove, ratio, num_unknown=num_unknown_labels)
                #fin = time.time()
                #print("get tensor done", fin - debut)
                C = np.transpose(tensor)
                #fin2 = time.time()
                #print("decomposition done", fin2 - fin)
                graph = embedding_matrix_2_kNN(C, k=num_nearest_neighbours).toarray()
                #fin3 = time.time()
                #print("KNN done", fin3 - fin2)
                # classe  b(i){> 0, < 0} means i ∈ {“+”, “-”}
                #print (graph.shape)
                beliefs = solve(graph, labels)
                #fin4 = time.time()
                #print("FaBP done", fin4 - fin3)
                # Compute hit rate
                hits = 0.
                compte=0.
                for l in range(len(beliefs)):
                    if labels[l]==0:
                        compte+=1
                        if beliefs[l] * all_labels[l] > 0:
                            hits += 1
                #print("return float belief", beliefs)
                beliefs[beliefs > 0] = 1
                beliefs[beliefs < 0] = -1
                #print("return int belief", beliefs)
                #print("labels correct", all_labels)
                #print("labels a completer", labels)
                #print("% Correct", hits/compte)
                acc.append(100*hits/compte)
            accuracy_mean[i,j]=np.mean(acc)
            accuracy_std[i, j] = np.std(acc)
    np.save('{}_ration_accuracy_val stats_mean'.format(k),accuracy_mean)
    np.save('{}_ration_accuracy_val stats_std'.format(k), accuracy_std)
print(time.time()-debut)