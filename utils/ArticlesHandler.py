from utils.decomposition import LDADecomposition
from utils.decomposition import ParafacDecomposition
from utils.decomposition import GloVeDecomposition
from utils.decomposition import TransformerDecomposition
from utils.ArticlesProvider import ArticlesProvider
from utils import Config
from utils.postprocessing import PostProcessing


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
        self.postprocessors = {}
        self.last_tensor = None

    def postprocess(self):
        for name, postprocessor in self.postprocessors.items():
            print('Running', name, 'postprocessor')
            postprocessor.apply(self.last_tensor)

    def get_tensor(self):
        if self.config.method_decomposition_embedding == 'GloVe':
            decomposition = GloVeDecomposition(self.config, self.articles)
        elif self.config.method_decomposition_embedding == 'Transformer':
            decomposition = TransformerDecomposition(self.config, self.articles)
        elif self.config.method_decomposition_embedding == 'LDA':
            decomposition = LDADecomposition(self.config, self.articles)
        else:  # parafac is default
            decomposition = ParafacDecomposition(self.config, self.articles)

        self.last_tensor = decomposition.apply()
        return self.last_tensor

    def add_postprocessing(self, postprocessor: PostProcessing, name=None):
        """
        Add a new postprocessing or update already existing
        """
        if name is None:
            name = "_internal_" + str(len(self.postprocessors.keys()))
        self.postprocessors[name] = postprocessor

