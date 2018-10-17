import numpy as np
import os
import json


def load_config():
    config_file = os.path.abspath(os.path.join(os.curdir, 'config.json'))
    with open(config_file, 'r', ) as config_file:
        config_string = config_file.read()

    return json.loads(config_string)


def solve(graph, labels):
    """
    :param graph: le graphe des plus proches voisins
    :param labels: labels incomplets {-1,0,1}
    :return: labels complets {-1,1}
    """
    identity = np.eye(len(graph))
    d_vec = np.sum(graph, 1)
    D = np.diag(d_vec)
    homophily = FaBP(D)
    c = (2 * homophily) / (1 - 4 * homophily * homophily)
    a = 2 * homophily * c
    M = identity + a * D - c * graph
    return np.linalg.solve(M, labels)


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


def get_rate(beliefs,labels,all_labels):
    # Compute hit rate
    TP = 0.
    TN = 0
    FP = 0
    FN = 0
    compte = 0
    for l in range(len(beliefs)):
        if labels[l] == 0:
            compte = compte + 1
            if beliefs[l] ==1 and all_labels[l] == 1:
                TP += 1
            if beliefs[l] ==-1 and all_labels[l] == -1:
                TN += 1
            if beliefs[l] ==1 and all_labels[l] ==-1:
                FP += 1
            if beliefs[l] ==-1 and all_labels[l] == 1:
                FN += 1
    return (TP/compte,TN/compte,FP/compte,FN/compte)

def accuracy(TP,TN,FP,FN):
    if TP+TN+FP+FN==0:
        return ((TP + TN) / (TP + TN + FP + FN + 1))
    else:
        return((TP+TN)/(TP+TN+FP+FN))

def precision(TP,FP):
    if TP + FP ==0:
        return ((TP ) / (TP + FP+1))
    else:
        return ((TP ) / (TP + FP))

def recall(TP,FN):
    if TP + FN == 0:
        return ((TP) / (TP + FN+1))
    else:
        return((TP)/(TP + FN))

def f1_score(prec,rec):
    if (prec + rec)==0:
        return (2 * (prec * rec) / (prec + rec+1))
    else:
        return(2*(prec*rec)/(prec+rec))
