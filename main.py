from ArticleTensor import ArticleTensor
from kNN import embedding_matrix_2_kNN
from utils import solve
import time

num_nearest_neighbours = 5
rank_parafac_decomposition = 10
num_fake_articles = 10
num_real_articles = 10
num_unknown_labels = 16
size_word_co_occurrence_window = 5
use_frequency = False
vocab_size = -1

assert num_fake_articles + num_real_articles > num_nearest_neighbours, "Can't have more neighbours than nodes!"

debut = time.time()
articleTensor = ArticleTensor('../Dataset/fakenewsdata1/Public Data')
articleTensor.get_articles('Buzzfeed Political News Dataset', number_fake=num_fake_articles,
                           number_real=num_real_articles)
articleTensor.build_word_to_index(max_words=vocab_size)
tensor, labels, all_labels = articleTensor.get_tensor(window=size_word_co_occurrence_window,
                                                      num_unknown=num_unknown_labels,
                                                      use_frequency=use_frequency)
# print(tensor.todense().dtype)
fin = time.time()

print("get tensor done", fin - debut)
_, (_, _, C) = ArticleTensor.get_parafac_decomposition(tensor, rank=rank_parafac_decomposition)
print(type(C))
# result = parafac_decomposition(tensor, rank_parafac_decomposition)
fin2 = time.time()
print("decomposition done", fin2 - fin)
graph = embedding_matrix_2_kNN(C, k=num_nearest_neighbours).toarray()
fin3 = time.time()
print("KNN done", fin3 - fin2)
# classe  b(i){> 0, < 0} means i ∈ {“+”, “-”}
beliefs = solve(graph, labels)
fin4 = time.time()
print("FaBP done", fin4 - fin3)

# Compute hit rate
hits = 0.
for i in range(len(beliefs)):
    if beliefs[i] * all_labels[i] >= 0:
        hits += 1


print("return float belief", beliefs)
beliefs[beliefs > 0] = 1
beliefs[beliefs < 0] = -1
print("return int belief", beliefs)
print("labels correct", all_labels)
print("labels to complete", labels)
print("% Correct", hits/len(all_labels))
