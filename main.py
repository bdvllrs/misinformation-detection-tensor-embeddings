from preprocessing import ArticlesProvider
from utils import solve, embedding_matrix_2_kNN, load_config, get_rate,accuracy,precision,recall,f1_score
import time
import numpy as np

config = load_config()

assert config['num_fake_articles'] + config['num_real_articles'] > config[
    'num_nearest_neighbours'], "Can't have more neighbours than nodes!"

debut = time.time()
articles = ArticlesProvider(config).setup()

C, labels, all_labels = articles.get_tensor(proportion_true_fake_label=config["proportion_true_fake_label"])
print(C,labels)
C, labels, all_labels = list(
        zip(*np.random.permutation(list(zip(C, labels, all_labels)))))
print(C,labels)

# print(tensor.todense().dtype)
fin = time.time()
print("get tensor and decomposition done", fin - debut)
graph = embedding_matrix_2_kNN(C, k=config['num_nearest_neighbours']).toarray()
fin3 = time.time()
print("KNN done", fin3 - fin)
# classe  b(i){> 0, < 0} means i ∈ {“+”, “-”}
beliefs = solve(graph, labels)
fin4 = time.time()
print("FaBP done", fin4 - fin3)

# Compute hit rate
print("return float belief", beliefs)
beliefs[beliefs > 0] = 1
beliefs[beliefs < 0] = -1

TP, TN, FP, FN=get_rate(beliefs,labels, all_labels)
acc = accuracy(TP, TN, FP, FN)
prec = precision(TP, FP)
rec = recall(TP, FN)
f1=f1_score( prec,rec)
print("return int belief", beliefs)
print("labels correct", all_labels)
print("labels to complete", labels)
print("% Correct (accuracy, precision, recall, f1_score)", 100*acc,prec*100,rec*100,f1*100)
