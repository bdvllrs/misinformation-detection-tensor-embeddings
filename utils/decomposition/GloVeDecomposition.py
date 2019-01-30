from utils import Config, load_glove_model
from utils.ArticlesProvider import ArticlesProvider
from utils.decomposition import Decomposition
import numpy as np
import torch


class GloVeDecomposition(Decomposition):
    def __init__(self, config: Config, articles: ArticlesProvider):
        super().__init__(config, articles)
        self.RNN = torch.nn.GRUCell(100, 100)
        self.glove = load_glove_model(config.paths.GloVe_adress)

    def apply(self):
        tensor = self.get_tensor_Glove(self.config.embedding.method_embedding_glove,
                                       self.config.embedding.vocab_util_pourcentage)
        return np.transpose(tensor)

    def get_tensor_Glove(self, method_embedding_glove, ratio):
        tensor = np.zeros((100, len(self.articles.article_list)))
        for k, article in enumerate(self.articles.article_list):
            tensor[:, k] = self.get_glove_matrix(article, ratio, method=method_embedding_glove)
        return tensor

    def get_glove_matrix(self, article, ratio, method="mean"):
        """
        Get the Glove of an article
        :param article
        """
        N = 0
        vector = np.zeros(100)
        vector_rnn = np.zeros((len(article), 1, 100))
        for k, word in enumerate(article):
            if word in self.articles.vocabulary and len(self.articles.frequency[word]) < (
                    ratio * self.articles.nb_all_articles):
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
            if not N:
                return vector
            return vector / N
