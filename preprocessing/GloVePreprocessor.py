from utils import Config, load_glove_model
from preprocessing import ArticlesProvider
from preprocessing.Preprocessor import Preprocessor
import numpy as np
import torch


class GloVePreprocessor(Preprocessor):
    def __init__(self, config: Config, articles: ArticlesProvider):
        super().__init__(config, articles)
        self.RNN = torch.nn.GRUCell(100, 100)
        self.glove = load_glove_model(config.GloVe_adress)

    def preprocess(self):
        tensor, labels, all_labels = self.get_tensor_Glove(self.config.method_embedding_glove,
                                                           self.config.vocab_util_pourcentage,
                                                           self.config.num_unknown_labels,
                                                           self.config.proportion_true_fake_label)
        return np.transpose(tensor), labels, all_labels

    def get_tensor_Glove(self, method_embedding_glove, ratio, num_unknown, proportion_true_fake_label=0.5):
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
        tensor = np.zeros((100, len(articles)))
        for k, article in enumerate(articles):
            tensor[:, k] = self.get_glove_matrix(article, ratio, method=method_embedding_glove)
        return tensor, labels, labels_untouched

    def get_glove_matrix(self, article, ratio, method="mean"):
        """
        Get the Glove of an article
        :param article
        """
        N = 0
        vector = np.zeros(100)
        vector_rnn = np.zeros((len(article), 1, 100))
        for k, word in enumerate(article):
            if word in self.articles.vocabulary and len(self.articles.frequency[word]) < (ratio * self.articles.nb_all_articles):
                if method == "mean":
                    try:
                        N += 1
                        vector = vector + self.glove[word]
                    except Exception:
                        vector = vector + self.glove['unk']
                if method == "RNN":
                    try:
                        N += 1
                        vector_rnn[k, :, :] = self.glove[word]
                    except Exception:
                        vector_rnn[k, :, :] = self.glove['unk']
        # print("Nombre de mots considéré en pourcentage", float(N) / float(len(article)))
        if method == "RNN":
            hx = torch.zeros(1, 100)
            for i in range(len(article)):
                hx = self.RNN(torch.from_numpy(vector_rnn[i]).float(), hx)
            vector = hx[0].detach().numpy()
            return vector
        else:
            return vector / N
