import numpy as np
from sklearn.neighbors import kneighbors_graph


def embedding_matrix_2_kNN(X, k, mode='connectivity'):
    """
    :param X: the embedding  matrix called C in the paper
    :param k: number of nearest neighbours
    :return: graph of the nearest neighbours

    :Example:

    >>> X = [[1, 1], [5, 3], [6, 4],[5, 4]]
    >>> embedding_matrix_2_kNN(X, 2).toarray()
    array([[0., 1., 0., 1.],
           [1., 0., 1., 1.],
           [0., 1., 0., 1.],
           [1., 1., 1., 0.]])

    """
    KNN_nonsym = kneighbors_graph(X, k, mode=mode, include_self=False)
    KNN = KNN_nonsym + np.transpose(KNN_nonsym)
    KNN[KNN == 2] = 1
    return KNN


def knn_similarities(articles, k):
    """
    :param articles: articles of nlp obj of spacy
    :param k: number of neighboors
    :return:
    """
    similarities = np.array([[art1.similarity(art2) for art1 in articles] for art2 in articles])
    index_higest_similarities = np.argpartition(similarities, k, axis=1)[:, :k]
    adj_mat = np.zeros((len(articles), len(articles)), dtype=int)
    for i, neightbors in enumerate(index_higest_similarities):
        adj_mat[i, neightbors] = 1
    # Force symetric
    adj_mat = adj_mat + np.transpose(adj_mat)
    adj_mat[adj_mat == 2] = 1
    return adj_mat
