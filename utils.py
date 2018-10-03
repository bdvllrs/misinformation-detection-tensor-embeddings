import numpy as np


def solve(graph, labels, homophily):
    identity = np.eye(len(graph))
    c = (2 * homophily) / (1 - 4 * homophily * homophily)
    a = 2 * homophily * c
    d_vec = np.sum(graph, 1)
    D = np.diag(d_vec)
    M = identity + a * D - c * graph
    return np.linalg.solve(M, labels)
