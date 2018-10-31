from numpy import transpose
from preprocessing import ArticleTensor


class ArticlesProvider:
    """
    Acts as a provider for ArticleTensor methods.
    It reduces the number of necessary params by deducing them from the config file.
    Also chooses which method to use to compute the tensor.
    """

    def __init__(self, config: dict):
        """
        :param config: Config dictionary
        :type config: dict
        """
        self.config = config
        self.articleTensor = ArticleTensor(config)

    def get_config(self, name: str, value: any):
        return self.config[name] if value is None else value

    def setup(self):
        self.get_articles()
        self.build_word_to_index()
        return self

    def get_articles(self, articles_directory: str = None, number_fake: int = None, number_real: int = None):
        articles_directory = self.get_config('dataset_name', articles_directory)
        number_fake = self.get_config('num_fake_articles', number_fake)
        number_real = self.get_config('num_real_articles', number_real)
        self.articleTensor.get_articles(articles_directory, number_fake, number_real)

    def build_word_to_index(self, max_words: int = None):
        max_words = self.get_config('vocab_size', max_words)
        self.articleTensor.build_word_to_index(max_words=max_words)

    def get_tensor(self, method: str = None, window: int = None, num_unknown: int = None, ratio: float = None,
                   use_frequency: bool = None, parafac_rank: int = None, method_embedding_glove: str = None,proportion_true_fake_label=0.5):
        method = self.get_config('method_decomposition_embedding', method)
        window = self.get_config('size_word_co_occurrence_window', window)
        num_unknown = self.get_config('num_unknown_labels', num_unknown)
        ratio = self.get_config('vocab_util_pourcentage', ratio)
        use_frequency = self.get_config('use_frequency', use_frequency)
        parafac_rank = self.get_config('rank_parafac_decomposition', parafac_rank)
        method_embedding_glove = self.get_config('method_embedding_glove', method_embedding_glove)
        if method == "decomposition":
            tensor, labels, all_labels = self.articleTensor.get_tensor_coocurrence(window, num_unknown, ratio,
                                                                                   use_frequency,
                                                                                   proportion_true_fake_label=proportion_true_fake_label)
            return self.articleTensor.get_parafac_decomposition(tensor, rank=parafac_rank)[1][2], labels, all_labels
        elif method == "GloVe":
            tensor, labels, all_labels = self.articleTensor.get_tensor_Glove(method_embedding_glove,
                                                                             ratio,
                                                                             num_unknown=num_unknown,
                                                                             proportion_true_fake_label=proportion_true_fake_label)
            return transpose(tensor), labels, all_labels
