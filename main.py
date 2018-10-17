from ArticleTensor import ArticleTensor
from utils import solve, embedding_matrix_2_kNN, load_config
import time
import numpy as np

config = load_config()

assert config['num_fake_articles'] + config['num_real_articles'] > config[
    'num_nearest_neighbours'], "Can't have more neighbours than nodes!"

debut = time.time()
articleTensor = ArticleTensor(config['dataset_path'])
articleTensor.get_articles(config['dataset_name'], number_fake=config['num_fake_articles'],
                           number_real=config['num_real_articles'])
articleTensor.build_word_to_index(max_words=config['vocab_size'])

if config["method_decomposition_embedding"]=="decomposition":
    tensor, labels, all_labels = articleTensor.get_tensor_coocurrence(window=config['size_word_co_occurrence_window'],
                                                                      num_unknown=config['num_unknown_labels'],
                                                                      ratio=config["vocab_util_pourcentage"],
                                                                      use_frequency=config['use_frequency'])
    _, (_, _, C) = ArticleTensor.get_parafac_decomposition(tensor, rank=config['rank_parafac_decomposition'])
if config["method_decomposition_embedding"]=="GloVe":
    tensor, labels, all_labels = articleTensor.get_tensor_Glove(config["method_embedding_glove"],
                                                                config["vocab_util_pourcentage"],
                                                                num_unknown=config['num_unknown_labels'])
    C = np.transpose(tensor)
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
print("% Correct", hits / len(all_labels))
