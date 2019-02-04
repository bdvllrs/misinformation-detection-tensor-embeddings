from utils.ArticlesHandler import ArticlesHandler
from utils import solve, embedding_matrix_2_kNN, get_rate, accuracy, precision, recall, f1_score
from utils.Trainer_graph import TrainerGraph
from utils import Config, accuracy_sentence_based
import time
import numpy as np
# from utils.postprocessing.SelectLabelsPostprocessor import SelectLabelsPostprocessor
from utils.Trainer_graph import TrainerGraph
from sklearn.metrics import accuracy_score

config = Config('config/')

debut = time.time()
handler = ArticlesHandler(config)

# Save in a pickle file. To open, use the pickle dataloader.
#handler.articles.save("../Dataset/train_fake.pkl")
# Only recompute labels:
# handler.articles.compute_labels()

C = handler.get_tensor()
# select_labels = SelectLabelsPostprocessor(config, handler.articles)
# handler.add_postprocessing(select_labels, "label-selection")
# handler.postprocess()
labels = handler.articles.labels
all_labels = np.array(handler.articles.labels_untouched)

if config.learning.method_learning == "FaBP":
    assert max(labels) == 2, "FaBP can only be used for binary classification."

print(len(all_labels), "Articles")

if config.graph.node_features == config.embedding.method_decomposition_embedding:
    C_nodes = C.copy()
else:
    config.embedding.set("method_decomposition_embedding", config.graph.method_create_graph)
    C_nodes = handler.get_tensor()

fin = time.time()
print("get tensor and decomposition done", fin - debut)
sentence_to_articles = None if not config.graph.sentence_based else handler.articles.sentence_to_article
graph = embedding_matrix_2_kNN(C, k=config.graph.num_nearest_neighbours,
                               sentence_to_articles=sentence_to_articles).toarray()
fin3 = time.time()
print("KNN done", fin3 - fin)

if config.learning.method_learning == "FaBP":
    # classe  b(i){> 0, < 0} means i ∈ {“+”, “-”}
    l = np.array(labels)
    beliefs = solve(graph, l)
    fin4 = time.time()
    print("FaBP done", fin4 - fin3)
else:
    #idx = np.random.randint(1,100,40)
    #labels= np.array(labels)
    #labels[idx] = 3
    #all_labels[idx] = 3
    trainer = TrainerGraph(C_nodes, graph, all_labels, labels)
    beliefs, acc_test = trainer.train()
    print(acc_test)
    fin4 = time.time()
    print("Learning done", fin4 - fin3)
    # Compute hit rate
    # TODO: changer pour le multiclasse...
    #beliefs[beliefs >= 0] = 1
    #beliefs[beliefs < 0] = 2


if config.graph.sentence_based:
    acc = accuracy_sentence_based(handler, beliefs)
else:
    print(all_labels, beliefs)
    acc = accuracy_score(all_labels, beliefs)

print("Accuracy", acc)
