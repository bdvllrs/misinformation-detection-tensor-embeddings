import numpy as np
from utils import get_fullpath, Config
from utils.dataloaders import FolderLoader, CSVLoader, PickleLoader, StatementsLoader
import pickle


class ArticlesProvider:
    def __init__(self, config: Config):
        """
        :param config: config dictionary
        :type config: dict
        """
        self.config = config
        self.path = config.dataset.dataset_path
        self.nb_all_articles = 0
        self.vocabulary = {}
        self.index_to_words = []
        self.frequency = {}  # dict : keys Words et values : list of files where the words are from
        self.words_to_index = {}
        self.articles = {}
        self.original_articles = {}
        self.article_list = []
        self.sentence_to_article = []
        self.index_to_label = {}
        self.labels = []
        self.statements = {}
        self.ordiginal_statements = {}
        self.labels_untouched = []
        self.nb_all_articles = 0
        self.dataloader = self.get_dataloader()
        self.load_articles()
        self.compute_labels()

    def save(self, path):
        """
        Save the data in a pickle file
        :return:
        """
        with open(get_fullpath(path), "wb") as file:
            to_picle = {"articles": self.articles, "original_articles": self.original_articles,
                        "vocabulary": self.vocabulary, "frequency": self.frequency}
            pickle.dump(to_picle, file)

    def get_dataloader(self):
        if self.config.dataset.type == "csv":
            return CSVLoader(self.config)
        elif self.config.dataset.type == "pickle":
            return PickleLoader(self.config)
        return FolderLoader(self.config)

    def load_articles(self):
        self.articles, self.original_articles, self.vocabulary, self.frequency = self.dataloader.load()
        if self.config.graph.sentence_based:
            statement_loader = StatementsLoader(self.config)
            statement_loader.frequency = self.frequency
            statement_loader.vocabulary = self.vocabulary
            self.statements, self.ordiginal_statements, self.vocabulary, self.fr = statement_loader.load()
        self._build_word_to_index()
        self.nb_all_articles = sum([len(self.articles[label]) for label in self.articles.keys()])

    def _build_word_to_index(self, in_freq_order=True, max_words=-1):
        """
        Build the index_to_word and word_to_index list and dict
        :param max_words: number max of words in vocab (only the most common ones) default, all of the vocab is kept.
        :param in_freq_order: if True, list in in order of appearance frequency.
        """
        if in_freq_order:
            vocab = sorted(list(self.vocabulary.items()), key=lambda x: x[1], reverse=True)
        else:
            vocab = list(self.vocabulary.items())
        max_words = max_words - 1 if max_words > 0 else -1
        vocab = vocab[:max_words]
        # Add <unk> to vocabulary
        vocab.append(('<unk>', 0))
        self.index_to_words, frequencies = list(zip(*vocab))
        self.index_to_words = list(self.index_to_words)
        self.words_to_index = {word: index for index, word in enumerate(self.index_to_words)}

    def get_word_index(self, word):
        """
        Returns the index of a word if known, the one of <unk> otherwise
        """
        if word in self.index_to_words:
            return self.words_to_index[word]
        return self.words_to_index['<unk>']

    def compute_labels(self):
        ratio_labeled = self.config.stats.ratio_labeled
        articles = []
        labels = []
        k = 0
        i = 0
        if self.config.graph.sentence_based:
            for label, a in self.statements.items():
                self.index_to_label[k + 1] = label
                a = list(map(lambda x: x['content'], a))
                self.sentence_to_article.append(i)
                articles.extend(a)
                labels.extend([k + 1] * len(a))
                k += 1
                i += 1
            start_label_to_remove = k
        for label, a in self.articles.items():
            self.index_to_label[k + 1] = label
            all_a = list(map(lambda x: x['content'], a))
            if self.config.graph.sentence_based:
                for all_s in all_a:
                    self.sentence_to_article.extend([i] * len(all_s))
                    articles.extend(all_s)
                    labels.extend([k + 1] * len(all_s))
                    i += 1
            else:
                self.sentence_to_article.extend([i] * len(all_a))
                articles.extend(all_a)
                labels.extend([k + 1] * len(all_a))
            k += 1
        # Shuffle the labels and articles
        articles, labels, self.sentence_to_article = list(
            zip(*np.random.permutation(list(zip(articles, labels, self.sentence_to_article)))))
        self.sentence_to_article = list(self.sentence_to_article)
        labels = list(labels)
        labels_untouched = labels[:]
        # Add zeros randomly to some labels
        if not self.config.graph.sentence_based:
            num_unknown = int(len(labels) * (1 - ratio_labeled) / len(self.articles.keys()))
            num_per_class = [num_unknown] * len(self.articles.keys())
            for k in range(len(labels)):
                if num_per_class[labels[k] - 1] > 0:
                    num_per_class[labels[k] - 1] -= 1
                    labels[k] = 0
        else:
            for k in range(len(labels)):
                if labels[k] > start_label_to_remove:  # only keep labels of the statements
                    labels[k] = 0
        self.labels = labels
        self.labels_untouched = labels_untouched
        self.article_list = articles
