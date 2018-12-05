import os
import numpy as np
import nltk
from utils import get_fullpath, Config


class ArticlesProvider:
    def __init__(self, config: Config):
        """
        :param config: config dictionary
        :type config: dict
        """
        self.config = config
        self.path = config.dataset_path
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
        self.load_articles(self.config.dataset_name, self.config.num_fake_articles, self.config.num_real_articles)
        self._build_word_to_index()
        self.compute_labels(self.config.num_unknown_labels, self.config.proportion_true_fake_label)

    def _get_content(self, filename: str, type: str = 'fake'):
        """
        Get the content of a given file
        :param filename: path to file to open
        """
        ps = nltk.PorterStemmer()
        with open(filename, 'r', encoding="utf-8", errors='ignore') as document:
            content = document.read().replace('\n', '').replace('\r', '').replace("\\'", "'")
        self.original_articles[type].append(content)
        content_words_tokenized = nltk.word_tokenize(content.lower())
        # Add words in the vocab

        for k, word in enumerate(content_words_tokenized):
            stemmed_word = ps.stem(word)
            # stemmed_word = word
            self.vocabulary[stemmed_word] = 1 if stemmed_word not in self.vocabulary.keys() else self.vocabulary[
                                                                                                     stemmed_word] + 1
            content_words_tokenized[k] = stemmed_word
            if stemmed_word not in self.frequency.keys():
                self.frequency[stemmed_word] = [filename]
            else:
                if filename not in self.frequency[stemmed_word]:
                    self.frequency[stemmed_word].append(filename)
        return content_words_tokenized

    def load_articles(self, articles_directory, number_fake, number_real):
        self.nb_all_articles = number_fake + number_real
        files_path_fake = get_fullpath(self.path, articles_directory, 'Fake')
        files_path_fake_titles = get_fullpath(self.path, articles_directory, 'Fake_titles')
        files_path_real = get_fullpath(self.path, articles_directory, 'Real')
        files_path_real_titles = get_fullpath(self.path, articles_directory, 'Real_titles')
        files_fake = np.random.choice(os.listdir(files_path_fake), number_fake)  # Get all files in the fake directory
        files_real = np.random.choice(os.listdir(files_path_real), number_real)  # Get all files in the real directory
        for file in files_fake:
            self.articles['fake'].append({
                'content': self._get_content(get_fullpath(files_path_fake, file), type='fake'),
                'title': self._get_content(get_fullpath(files_path_fake_titles, file), type='fake')
            })
        for file in files_real:
            self.articles['real'].append({
                'content': self._get_content(get_fullpath(files_path_real, file), type='real'),
                'title': self._get_content(get_fullpath(files_path_real_titles, file), type='real')
            })

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

    def compute_labels(self, num_unknown, proportion_true_fake_label):
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
