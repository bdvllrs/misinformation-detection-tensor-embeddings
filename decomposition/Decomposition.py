from utils.ArticlesProvider import ArticlesProvider
from utils import Config


class Decomposition:
    def __init__(self, config: Config, articles: ArticlesProvider):
        self.config = config
        self.articles = articles

    def preprocess(self):
        """
        :return: tensor, found labels, ground truth labels
        """
        raise NotImplementedError
