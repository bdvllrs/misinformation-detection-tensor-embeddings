from utils.dataloaders import DataLoader
import nltk
import os
from utils import get_fullpath, Config
import numpy as np


class FolderLoader(DataLoader):
    def __init__(self, config: Config):
        super().__init__(config)
        self.articles = {
            'fake': [],
            'real': []
        }
        self.original_articles = {'fake': [], 'real': []}
        self.vocabulary = {}
        self.frequency = {}  # dict : keys Words et values : list of files where the words are from

    def load(self):
        self.load_articles(self.config.dataset_path, self.config.dataset_name,
                           self.config.num_fake_articles, self.config.num_real_articles)
        return self.articles, self.original_articles, self.vocabulary, self.frequency

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

    def load_articles(self, path, articles_directory, number_fake, number_real):
        files_path_fake = get_fullpath(path, articles_directory, 'Fake')
        files_path_fake_titles = get_fullpath(path, articles_directory, 'Fake_titles')
        files_path_real = get_fullpath(path, articles_directory, 'Real')
        files_path_real_titles = get_fullpath(path, articles_directory, 'Real_titles')
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
