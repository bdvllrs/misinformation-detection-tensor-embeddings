import numpy as np


def solve(graph, labels):
    """

    :param graph: le graphe des plus proches voisins
    :param labels: labels incomplets {-1,0,1}
    :return: labels complets {-1,1}
    """
    identity = np.eye(len(graph))
    d_vec = np.sum(graph, 1)
    D = np.diag(d_vec)
    homophily=FaBP(D)
    c = (2 * homophily) / (1 - 4 * homophily * homophily)
    a = 2 * homophily * c
    M = identity + a * D - c * graph
    return np.linalg.solve(M, labels)


def FaBP(D):
    """
    Optmisation de la fonction BP
    :param D: D_ii=sum(graphe_jj)
    :return:homophily entre 0 et 1 mesure la connectivité des noeuds
    """
    c1=2+np.trace(D)
    c2=np.trace(np.power(D,2))-1
    h1=(1/(2+2*np.max(D)))
    h2=np.sqrt((-c1+np.sqrt(c1*c1+4*c2))/(8*c2))
    h=max(h1,h2)
    return(h)


"""
Example :

graphe=[[0. 1. 0. 1.]
 [1. 0. 1. 1.]
 [0. 1. 0. 1.]
 [1. 1. 1. 0.]]

D=  ([[2, 0, 0, 0],
       [0, 3, 0, 0],
       [0, 0, 2, 0],
       [0, 0, 0, 3]])

h=hypocitie= 0.13454551928275627


"""
