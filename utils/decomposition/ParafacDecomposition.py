from utils.decomposition import Decomposition
import sparse
from tensorly.contrib.sparse.decomposition import tucker


class ParafacDecomposition(Decomposition):
    def apply(self):
        tensor = self.get_tensor_coocurrence(self.config.embedding.size_word_co_occurrence_window,
                                             self.config.stats.num_unknown_labels,
                                             self.config.embedding.vocab_util_pourcentage)

        return self.get_parafac_decomposition(tensor, rank=self.config.embedding.rank_parafac_decomposition)[1][2]

    def get_tensor_coocurrence(self, window, ratio, use_frequency=True):
        coordinates = []
        data = []
        for k, article in enumerate(self.articles.article_list):
            coords, d = self.get_sparse_co_occurrence_matrix(article, window, k, ratio, use_frequency)
            coordinates.extend(coords)
            data.extend(d)
        coordinates = list(zip(*coordinates))
        tensor = sparse.COO(coordinates, data,
                            shape=(len(self.articles.index_to_words), len(self.articles.index_to_words),
                                   len(self.articles.article_list)))
        return tensor

    def get_sparse_co_occurrence_matrix(self, article, window, article_index, ratio, use_frequency=True):
        """
        Get the co occurrence matrix as sparse matrix of an article
        :param article_index: index of the corresponding article
        :param article:
        :param window: window to consider the words around
        :param use_frequency: if True, co occurrence matrix has the count with each other words else only a boolean
        """
        half_window = window // 2  # half to the right, half to the left
        coordinates = []
        data = []
        for k, word in enumerate(article):
            if word in self.articles.vocabulary and len(
                    self.articles.frequency[word]) < ratio * self.articles.nb_all_articles:
                neighbooring_words = (article[max(0, k - half_window): k] if k > 0 else []) + (
                    article[k + 1: min(len(article), k + 1 + half_window)] if k < len(article) - 1 else [])
                word_key = self.articles.get_word_index(word)
                for neighbooring_word in neighbooring_words:
                    coord = (word_key, self.articles.get_word_index(neighbooring_word), article_index)
                    if coord in coordinates and use_frequency:
                        data[coordinates.index(coord)] += 1.
                    else:
                        coordinates.append(coord)
                        data.append(1.)
        return coordinates, data

    @staticmethod
    def get_parafac_decomposition(tensor, rank):
        """
        Returns
        :param tensor:
        :param rank:
        :return: 3 matrix: (vocab, rank) (vocab, rank) and (num of articles, rank)
        """
        return tucker(tensor, rank=rank)
