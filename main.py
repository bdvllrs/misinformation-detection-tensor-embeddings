from utils.ArticlesHandler import ArticlesHandler
from utils import solve, embedding_matrix_2_kNN, get_rate, accuracy, precision, recall, f1_score
from utils import Config
import time
import numpy as np
from postprocessing.SelectLabelsPostprocessor import SelectLabelsPostprocessor
from utils.Trainer_graph import TrainerGraph

config = Config('config/')

assert config.stats.num_fake_articles + config.stats.num_real_articles > \
       config.graph.num_nearest_neighbours, "Can't have more neighbours than nodes!"

debut = time.time()
articles = ArticlesHandler(config)

C = articles.get_tensor()
select_labels = SelectLabelsPostprocessor(config, articles.articles)
articles.add_postprocessing(select_labels, "label-selection")
articles.postprocess()
labels = articles.articles.labels
all_labels = articles.articles.labels_untouched

C, labels, all_labels = list(
    zip(*np.random.permutation(list(zip(C, labels, all_labels)))))

fin = time.time()
print("get tensor and decomposition done", fin - debut)
graph = embedding_matrix_2_kNN(C, k=config.graph.num_nearest_neighbours).toarray()
fin3 = time.time()
print("KNN done", fin3 - fin)

if config.learning.method_learning == "FaPB":
    # classe  b(i){> 0, < 0} means i ∈ {“+”, “-”}
    beliefs = solve(graph, labels)
    fin4 = time.time()
    print("FaBP done", fin4 - fin3)
else:
    trainer = TrainerGraph(C, graph, all_labels, labels)
    beliefs = trainer.train()
    fin4 = time.time()
    print("Learning done", fin4 - fin3)

# Compute hit rate
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

