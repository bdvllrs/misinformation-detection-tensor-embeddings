from decomposition.ParafacDecomposition import ParafacDecomposition
from decomposition.GloVeDecomposition import GloVeDecomposition
from utils.ArticlesProvider import ArticlesProvider
from utils import Config
from preprocessing.Preprocessor import Preprocessor


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
        self.preprocessors = {}

    def setup(self):
        """
        Apply all preprocessing to the articles
        :return:
        """
        for preprocessor in self.preprocessors.values():
            preprocessor.preprocess()

    def get_tensor(self):
        if self.config.method_decomposition_embedding == 'parafac':
            decomposition = ParafacDecomposition(self.config, self.articles)
        else:  # elif self.config.method_decomposition_embedding == 'GloVe':
            decomposition = GloVeDecomposition(self.config, self.articles)
        return decomposition.preprocess()

    def add_preprocessing(self, preprocessor: Preprocessor, name=None):
        """
        Add a new preprocessing or update already existing
        """
        if name is None:
            name = "_internal_" + str(len(self.preprocessors.keys()))
        self.preprocessors[name] = preprocessor

