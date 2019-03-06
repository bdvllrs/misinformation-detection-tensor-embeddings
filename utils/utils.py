import statistics

import numpy as np
import os
import json


def load_config():
    config_file = os.path.abspath(os.path.join(os.curdir, 'config2.json'))
    with open(config_file, 'r', ) as config_file:
        config_string = config_file.read()

    return json.loads(config_string)


def solve(graph, labels):
    """
    :param graph: le graphe des plus proches voisins
    :param labels: labels incomplets {0,1, 2}
    :return: labels complets {1, 2}
    """
    labels[labels == 2] = -1  # set to {-1, 0, 1}
    identity = np.eye(len(graph))
    d_vec = np.sum(graph, 1)
    D = np.diag(d_vec)
    homophily = FaBP(D)
    c = (2 * homophily) / (1 - 4 * homophily * homophily)
    a = 2 * homophily * c
    M = identity + a * D - c * graph
    predicted_labels = np.linalg.solve(M, labels)
    predicted_labels[predicted_labels >= 0] = 1  # reset to {1, 2} instead of {-1, 1}
    predicted_labels[predicted_labels < 0] = 2  # reset to {1, 2} instead of {-1, 1}
    return predicted_labels


def FaBP(D):
    """
    Optmisation de la fonction BP
    :param D: D_ii=sum(graphe_jj)
    :return: homophily entre 0 et 1 mesure la connectivité des noeuds

    :Example:

    >>> FaBP([[2, 0, 0, 0], \
              [0, 3, 0, 0], \
              [0, 0, 2, 0], \
              [0, 0, 0, 3]])
    0.13454551928275627
    """
    c1 = 2 + np.trace(D)
    c2 = np.trace(np.power(D, 2)) - 1
    h1 = (1 / (2 + 2 * np.max(D)))
    h2 = np.sqrt((-c1 + np.sqrt(c1 * c1 + 4 * c2)) / (8 * c2))
    return max(h1, h2)


def get_rate(beliefs, labels, all_labels):
    # Compute hit rate
    #-1 devient 2
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    compte = 0
    for l in range(len(beliefs)):
        if labels[l] == 0:
            compte = compte + 1
            if beliefs[l] == 1 and all_labels[l] == 1:
                TP += 1
            if beliefs[l] == 2 and all_labels[l] == 2:
                TN += 1
            if beliefs[l] == 1 and all_labels[l] == 2:
                FP += 1
            if beliefs[l] == 2 and all_labels[l] == 1:
                FN += 1
    if compte ==0:
        compte = 1
    return (TP / compte, TN / compte, FP / compte, FN / compte)


def accuracy(TP, TN, FP, FN):
    if TP + TN + FP + FN == 0:
        return ((TP + TN) / (TP + TN + FP + FN + 1))
    else:
        return ((TP + TN) / (TP + TN + FP + FN))

def accuracy2(TP, TN, FP, FN):
    if TP + TN + FP + FN == 0:
        return ((TP + TN) / (TP + TN + FP + FN + 1))
    else:
        return ((TP + TN) / (TP + TN + FP + FN))


def precision(TP, FP):
    if TP + FP == 0:
        return ((TP) / (TP + FP + 1))
    else:
        return ((TP) / (TP + FP))


def recall(TP, FN):
    if TP + FN == 0:
        return ((TP) / (TP + FN + 1))
    else:
        return ((TP) / (TP + FN))


def f1_score(prec, rec):
    if (prec + rec) == 0:
        return (2 * (prec * rec) / (prec + rec + 1))
    else:
        return (2 * (prec * rec) / (prec + rec))


def get_fullpath(*path):
    """
    Returns an absolute path given a relative path
    """
    path = [os.path.curdir] + list(path)
    return os.path.abspath(os.path.join(*path))


def load_glove_model(glove_file):
    """
    :param glove_file: adress of glove file
    :return:
    """
    print("Loading Glove Model")
    f = open(glove_file, 'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model


def accuracy_sentence_based(handler, beliefs):
    fake_indexes = [k for (k, l) in handler.articles.index_to_label.items() if l == 'fake']
    real_indexes = [k for (k, l) in handler.articles.index_to_label.items() if l == 'real']

    all_labels = np.array(handler.articles.labels_untouched)

    beliefs[beliefs == max(fake_indexes)] = min(fake_indexes)
    beliefs[beliefs == max(real_indexes)] = min(real_indexes)
    all_labels[all_labels == max(fake_indexes)] = min(fake_indexes)
    all_labels[all_labels == max(real_indexes)] = min(real_indexes)

    beliefs_per_article = {}
    true_labels = {}
    for k in range(len(beliefs)):
        article_id = handler.articles.sentence_to_article[k]
        true_labels[article_id] = all_labels[k]
        if article_id not in beliefs_per_article:
            beliefs_per_article[article_id] = [beliefs[k]]
        else:
            beliefs_per_article[article_id].append(beliefs[k])

    num_good = 0
    for k in range(len(true_labels.keys())):
        if statistics.median(beliefs_per_article[k]) == true_labels[k]:
            num_good += 1

    return num_good / float(len(true_labels.keys()))
