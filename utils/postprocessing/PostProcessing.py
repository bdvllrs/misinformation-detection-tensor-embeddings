from utils import Config
from utils.ArticlesProvider import ArticlesProvider


class PostProcessing:
    def __init__(self, config: Config, articles: ArticlesProvider):
        self.config = config
        self.articles = articles

    def apply(self, tensor):
        raise NotImplementedError
