from preprocessing.Preprocessor import Preprocessor
import sparse
import numpy as np
from tensorly.contrib.sparse.decomposition import tucker


class ParafacPreprocessor(Preprocessor):
    def preprocess(self):
        tensor, labels, all_labels = self.get_tensor_coocurrence(self.config.size_word_co_occurrence_window,
                                                                 self.config.num_unknown_labels,
                                                                 self.config.vocab_util_pourcentage,
                                                                 self.config.use_frequency,
                                                                 self.config.proportion_true_fake_label)
        return self.get_parafac_decomposition(tensor, rank=self.config.rank_parafac_decomposition)[1][
                   2], labels, all_labels

    def get_tensor_coocurrence(self, window, num_unknown, ratio, use_frequency=True, proportion_true_fake_label=0.5):
        true_articles = [article['content'] for article in self.articles.articles['real']]
        fake_articles = [article['content'] for article in self.articles.articles['fake']]
        articles = true_articles + fake_articles
        labels = []
        for k in range(len(articles)):
            if k < len(self.articles.articles['fake']):
                labels.append(-1)
            else:
                labels.append(1)
        # Shuffle the labels and articles
        articles, labels = list(zip(*np.random.permutation(list(zip(articles, labels)))))
        labels = list(labels)
        labels_untouched = labels[:]
        # Add zeros randomly to some labels
        num_known = len(labels) - num_unknown
        number_true_unknown = len(true_articles) - int(proportion_true_fake_label * num_known)
        number_false_unknown = len(fake_articles) - (num_known - int(proportion_true_fake_label * num_known))
        for k in range(len(labels)):
            if (number_true_unknown > 0) & (labels[k] == 1):
                labels[k] = 0
                number_true_unknown -= 1
            if (labels[k] == -1) & (number_false_unknown > 0):
                labels[k] = 0
                number_false_unknown -= 1
        coordinates = []
        data = []
        for k, article in enumerate(articles):
            coords, d = self.get_sparse_co_occurrence_matrix(article, window, k, ratio, use_frequency)
            coordinates.extend(coords)
            data.extend(d)
        coordinates = list(zip(*coordinates))
        tensor = sparse.COO(coordinates, data,
                            shape=(len(self.articles.index_to_words), len(self.articles.index_to_words), len(articles)))
        return tensor, labels, labels_untouched

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
