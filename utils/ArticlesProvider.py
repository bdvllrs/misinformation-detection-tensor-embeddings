import os
import numpy as np
import nltk
from utils import get_fullpath, Config
from utils.dataloaders import FolderLoader, CSVLoader, PickleLoader
import pickle


class ArticlesProvider:
    def __init__(self, config: Config):
        """
        :param config: config dictionary
        :type config: dict
        """
        self.config = config
        self.path = config.Dataset.dataset_path
        self.nb_all_articles = 0
        self.vocabulary = {}
        self.index_to_words = []
        self.frequency = {}  # dict : keys Words et values : list of files where the words are from
        self.words_to_index = {}
        self.articles = {
            'fake': [],
            'real': []
        }
        self.original_articles = {'fake': [], 'real': []}
        self.article_list = []
        self.labels = []
        self.labels_untouched = []
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
        if self.config.Dataset.type == "csv":
            return CSVLoader(self.config)
        elif self.config.Dataset.type == "pickle":
            return PickleLoader(self.config)
        return FolderLoader(self.config)

    def load_articles(self):
        self.articles, self.original_articles, self.vocabulary, self.frequency = self.dataloader.load()
        self._build_word_to_index()

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
        num_unknown, proportion_true_fake_label = self.config.Stats.num_unknown_labels, self.config.Stats.proportion_true_fake_label
        true_articles = [article['content'] for article in self.articles['real']]
        fake_articles = [article['content'] for article in self.articles['fake']]
        articles = true_articles + fake_articles
        labels = []
        for k in range(len(articles)):
            if k < len(self.articles['fake']):
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
        self.labels = labels
        self.labels_untouched = labels_untouched
        self.article_list = articles
