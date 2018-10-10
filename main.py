from ArticleTensor import ArticleTensor
from kNN import embedding_matrix_2_kNN
from utils import solve
import time
import numpy as np

ratio=0.8
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
tensor, labels, all_labels = articleTensor.get_tensor(method_embedding_glove, ratio, num_unknown=num_unknown_labels)
fin = time.time()
print("get tensor done", fin - debut)
C = np.transpose(tensor)
fin2 = time.time()
print("decomposition done", fin2 - fin)
graph = embedding_matrix_2_kNN(C, k=num_nearest_neighbours).toarray()
fin3 = time.time()
print("KNN done", fin3 - fin2)
# classe  b(i){> 0, < 0} means i ∈ {“+”, “-”}
print (graph.shape)
beliefs = solve(graph, labels)
fin4 = time.time()
print("FaBP done", fin4 - fin3)
# Compute hit rate
hits = 0.
compte=0.
for l in range(len(beliefs)):
    if labels[l]==0:
        compte+=1
        if beliefs[l] * all_labels[l] > 0:
            hits += 1
print("return float belief", beliefs)
beliefs[beliefs > 0] = 1
beliefs[beliefs < 0] = -1
print("return int belief", beliefs)
print("labels correct", all_labels)
print("labels a completer", labels)
print("% Correct", hits/compte)
