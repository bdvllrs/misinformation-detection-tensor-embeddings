from preprocessing.Preprocessor import Preprocessor
from utils.Config import Config
from utils.ArticlesProvider import ArticlesProvider


class SelectLabelsPreprocessor(Preprocessor):
    def __init__(self, config: Config, articles: ArticlesProvider):
        super().__init__(config, articles)

    """
    Select smartly the articles to label to optimize the result
    """
    def preprocess(self):
        graph = embedding_matrix_2_kNN(C, k=config.num_nearest_neighbours).toarray()

