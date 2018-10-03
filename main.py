from ArticleTensor import ArticleTensor
from kNN import embedding_matrix_2_kNN
from utils import solve

num_nearest_neighbours = 5
rank_parafac_decomposition = 10
num_fake_articles = 10
num_real_articles = 10
num_unknown_labels = 10
size_word_co_occurrence_window = 5
use_frequency = False
homophily = 1

assert num_fake_articles + num_real_articles > num_nearest_neighbours, "Can't have more neighbours than nodes!"
assert 0 <= homophily <= 1, "Homophily should between 0 and 1."

articleTensor = ArticleTensor('../Dataset/fakenewsdata1/Public Data')
articleTensor.get_articles('Buzzfeed Political News Dataset', number_fake=num_fake_articles,
                           number_real=num_real_articles)
articleTensor.build_word_to_index()
tensor, labels, all_labels = articleTensor.get_tensor(window=size_word_co_occurrence_window, num_unknown=num_unknown_labels,
                                          use_frequency=use_frequency)
_, _, C = ArticleTensor.get_parafac_decomposition(tensor, rank=rank_parafac_decomposition)

graph = embedding_matrix_2_kNN(C, k=num_nearest_neighbours).toarray()

beliefs = solve(graph, labels, homophily)

print(beliefs)
print(all_labels)



