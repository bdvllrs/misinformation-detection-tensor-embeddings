from preprocessing.ParafacPreprocessor import ParafacPreprocessor
from preprocessing.GloVePreprocessor import GloVePreprocessor
from preprocessing.ArticlesProvider import ArticlesProvider
from utils import Config


class ArticlesHandler:
    """
    Acts as a provider for ArticleTensor methods.
    It reduces the number of necessary params by deducing them from the config file.
    Also chooses which method to use to compute the tensor.
    """

    def __init__(self, config: Config):
        """
        :param config: Config dictionary
        :type config: dict
        """
        self.config = config
        self.articles = ArticlesProvider(config)

    def get_tensor(self):
        if self.config.method_decomposition_embedding == 'decomposition':
            preprocessor = ParafacPreprocessor(self.config, self.articles)
        else:  # elif self.config.method_decomposition_embedding == 'GloVe':
            preprocessor = GloVePreprocessor(self.config, self.articles)
        return preprocessor.preprocess()

