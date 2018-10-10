import numpy as np
from sklearn.neighbors import kneighbors_graph
import time


def embedding_matrix_2_kNN(X, k):
    """

    :param X: the embedding  matrix called C in the paper
    :param k: number of nearest neighbours
    :return: graph of the nearest neighbours
    """
    KNN_nonsym = kneighbors_graph(X, k, mode='connectivity', include_self=False)
    KNN = KNN_nonsym + np.transpose(KNN_nonsym)
    KNN[KNN == 2] = 1
    return KNN


"""
#test :
X = [[1,1], [5,3], [6,4],[5,4]]
start=time.time()
Y=embedding_matrix_2_kNN(X,2)
end=time.time()

print('duree (s): ', end-start,)
print(Y.toarray())


Ground true :
0.055 (s)
[[0. 1. 0. 1.]
 [1. 0. 1. 1.]
 [0. 1. 0. 1.]
 [1. 1. 1. 0.]]
"""
