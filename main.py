from ArticleTensor import ArticleTensor
from kNN import embedding_matrix_2_kNN
from utils import solve

# import scipy.io

num_nearest_neighbours = 10
rank_parafac_decomposition = 10
num_fake_articles = 10
num_real_articles = 10
num_unknown_labels = 7
size_word_co_occurrence_window = 5
use_frequency = False
vocab_size = 1000

assert num_fake_articles + num_real_articles > num_nearest_neighbours, "Can't have more neighbours than nodes!"

articleTensor = ArticleTensor('../Dataset/fakenewsdata1/Public Data')
articleTensor.get_articles('Buzzfeed Political News Dataset', number_fake=num_fake_articles,
                           number_real=num_real_articles)
articleTensor.build_word_to_index(max_words=vocab_size)
tensor, labels, all_labels = articleTensor.get_tensor(window=size_word_co_occurrence_window,
                                                      num_unknown=num_unknown_labels,
                                                      use_frequency=use_frequency)

_, _, C = ArticleTensor.get_parafac_decomposition(tensor, rank=rank_parafac_decomposition)

graph = embedding_matrix_2_kNN(C, k=num_nearest_neighbours).toarray()

beliefs = solve(graph, labels)

# Compute hit rate
hits = 0.
for i in range(len(beliefs)):
    if beliefs[i] * all_labels[i] >= 0:
        hits += 1

print(beliefs)
print(all_labels)
print(hits)
