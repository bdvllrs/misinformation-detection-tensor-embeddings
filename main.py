from utils.ArticlesHandler import ArticlesHandler
from utils import solve, embedding_matrix_2_kNN, get_rate, accuracy2 as accuracy, precision, recall, f1_score
from utils import Config
import time
import numpy as np

config = Config(file='config')

assert config.num_fake_articles + config.num_real_articles > config.num_nearest_neighbours, "Can't have more neighbours than nodes!"

debut = time.time()
handler = ArticlesHandler(config)

C = handler.get_tensor()
# select_labels = SelectLabelsPostprocessor(config, handler.articles)
# handler.add_postprocessing(select_labels, "label-selection")
# handler.postprocess()
labels = handler.articles.labels
all_labels = handler.articles.labels_untouched

print(C, labels)
C, labels, all_labels = list(
    zip(*np.random.permutation(list(zip(C, labels, all_labels)))))
print(C, labels)

# print(tensor.todense().dtype)
fin = time.time()
print("get tensor and decomposition done", fin - debut)
graph = embedding_matrix_2_kNN(C, k=config.num_nearest_neighbours).toarray()
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

TP, TN, FP, FN = get_rate(beliefs, labels, all_labels)
acc = accuracy(TP, TN, FP, FN)
prec = precision(TP, FP)
rec = recall(TP, FN)
f1 = f1_score(prec, rec)
print("return int belief", beliefs)
print("labels correct", all_labels)
print("labels to complete", labels)
print("% Correct (accuracy, precision, recall, f1_score)", 100 * acc, prec * 100, rec * 100, f1 * 100)
print(100 * float(len(np.array(list(labels)) == 0.))/float(len(list(labels))), '% of labels')

