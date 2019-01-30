import json
import os
import re

import ftfy
import nltk
import numpy as np
import spacy
import sparse
import torch

from tensorly.contrib.sparse.decomposition import tucker
from transformer.model_pytorch import DEFAULT_CONFIG, TransformerModel, load_openai_pretrained_model
from utils import get_fullpath, load_glove_model


class ArticleTensor:
    def __init__(self, config: dict):
        """
        :param config: config dictionary
        :type config: dict
        """
        print(
            '\033[91m' + ' ArticleTensor is Deprecated! Please use ArticlesHandler and change values in the' +
            ' configuration class using config.set. This class is not maintained anymore.')
        self.config = config
        self.path = config['dataset_path']
        self.nbre_all_article = 0
        self.vocabulary = {}
        self.index_to_words = []
        self.RNN = torch.nn.GRUCell(100, 100)
        self.frequency = {}  # dictinnaire : clefs Words et attributs : liste de files dans lesquels ces mots sont
        self.words_to_index = {}
        self.articles = {
            'fake': [],
            'real': []
        }
        if config["method_decomposition_embedding"] == "GloVe":
            self.glove = load_glove_model(config["GloVe_adress"])

        if config["method_decomposition_embedding"] == "Transformer":
            self.encoder = json.load(open(config["encoder_path"]))
            merges = open(config["bpe_path"], encoding='utf-8').read().split('\n')[1:-1]
            merges = [tuple(merge.split()) for merge in merges]
            self.bpe_ranks = dict(zip(merges, range(len(merges))))
            self.bpe_ranks = dict(zip(merges, range(len(merges))))
            self.model = TransformerModel(DEFAULT_CONFIG)
            load_openai_pretrained_model(self.model,  path='./transformer/model/', path_names='./transformer')
            self.cache = {}

    def get_content(self, filename: str):
        """
        Get the content of a given file
        :param filename: path to file to open
        """
        ps = nltk.PorterStemmer()
        with open(filename, 'r', encoding="utf-8", errors='ignore') as document:
            content = document.read().replace('\n', '').replace('\r', '')
        content_words_tokenized = nltk.word_tokenize(content.lower())
        # Add words in the vocab

        for k, word in enumerate(content_words_tokenized):
            stemmed_word = ps.stem(word)
            self.vocabulary[stemmed_word] = 1 if stemmed_word not in self.vocabulary.keys() else self.vocabulary[
                                                                                                     stemmed_word] + 1
            content_words_tokenized[k] = stemmed_word
            if stemmed_word not in self.frequency.keys():
                self.frequency[stemmed_word] = [filename]
            else:
                if filename not in self.frequency[stemmed_word]:
                    self.frequency[stemmed_word].append(filename)
        return content_words_tokenized

    def get_articles(self, articles_directory, number_fake, number_real):
        self.nbre_all_article = number_fake + number_real
        files_path_fake = get_fullpath(self.path, articles_directory, 'Fake')
        files_path_fake_titles = get_fullpath(self.path, articles_directory, 'Fake_titles')
        files_path_real = get_fullpath(self.path, articles_directory, 'Real')
        files_path_real_titles = get_fullpath(self.path, articles_directory, 'Real_titles')
        files_fake = np.random.choice(os.listdir(files_path_fake), number_fake)  # Get all files in the fake directory
        files_real = np.random.choice(os.listdir(files_path_real), number_real)  # Get all files in the real directory
        for file in files_fake:
            self.articles['fake'].append({
                'content': self.get_content(get_fullpath(files_path_fake, file)),
                'title': self.get_content(get_fullpath(files_path_fake_titles, file))
            })
        for file in files_real:
            self.articles['real'].append({
                'content': self.get_content(get_fullpath(files_path_real, file)),
                'title': self.get_content(get_fullpath(files_path_real_titles, file))
            })

    def build_word_to_index(self, in_freq_order=True, max_words=-1):
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

    def get_glove_matrix(self, article, ratio, method="mean"):
        """
        Get the Glove of an article
        :param article
        """
        N = 0
        vector = np.zeros(100)
        vector_rnn = np.zeros((len(article), 1, 100))
        for k, word in enumerate(article):
            if word in self.vocabulary and len(self.frequency[word]) < (ratio * self.nbre_all_article):
                if method == "mean":
                    try:
                        N += 1
                        vector = vector + self.glove[word]
                    except Exception:
                        vector = vector + self.glove['unk']
                if method == "RNN":
                    try:
                        N += 1
                        vector_rnn[k, :, :] = self.glove[word]
                    except Exception:
                        vector_rnn[k, :, :] = self.glove['unk']
        # print("Nombre de mots considéré en pourcentage", float(N) / float(len(article)))
        if method == "RNN":
            hx = torch.zeros(1, 100)
            for i in range(len(article)):
                hx = self.RNN(torch.from_numpy(vector_rnn[i]).float(), hx)
            vector = hx[0].detach().numpy()
            return vector
        else:
            return vector / N

    def get_sparse_co_occurrence_matrix(self, article, window, article_index, ratio, use_frequency=True):
        """
        Get the co occurrence matrix as sparse matrix of an article
        :param article_index: index of the corresponding article
        :param article:
        :param window: window to consider the words around
        :param use_frequency: if True, co occurrence matrix has the count with each other words else only a boolean
        """
        half_window = window // 2  # half to the right, half to the left
        coordinates = []
        data = []
        for k, word in enumerate(article):
            if word in self.vocabulary and len(self.frequency[word]) < ratio * self.nbre_all_article:
                neighbooring_words = (article[max(0, k - half_window): k] if k > 0 else []) + (
                    article[k + 1: min(len(article), k + 1 + half_window)] if k < len(article) - 1 else [])
                word_key = self.get_word_index(word)
                for neighbooring_word in neighbooring_words:
                    coord = (word_key, self.get_word_index(neighbooring_word), article_index)
                    if coord in coordinates and use_frequency:
                        data[coordinates.index(coord)] += 1.
                    else:
                        coordinates.append(coord)
                        data.append(1.)
        return coordinates, data

    def get_tensor_coocurrence(self, window, num_unknown, ratio, use_frequency=True, proportion_true_fake_label=0.5):
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
        # articles, labels, labels_untouched = list(
        #    zip(*np.random.permutation(list(zip(articles, labels, labels_untouched)))))
        coordinates = []
        data = []
        for k, article in enumerate(articles):
            coords, d = self.get_sparse_co_occurrence_matrix(article, window, k, ratio, use_frequency)
            coordinates.extend(coords)
            data.extend(d)
        coordinates = list(zip(*coordinates))
        tensor = sparse.COO(coordinates, data,
                            shape=(len(self.index_to_words), len(self.index_to_words), len(articles)))
        return tensor, labels, labels_untouched

    def get_tensor_Glove(self, method_embedding_glove, ratio, num_unknown, proportion_true_fake_label=0.5):
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
        # articles, labels, labels_untouched = list(
        #    zip(*np.random.permutation(list(zip(articles, labels, labels_untouched)))))
        tensor = np.zeros((100, len(articles)))
        for k, article in enumerate(articles):
            tensor[:, k] = self.get_glove_matrix(article, ratio, method=method_embedding_glove)
        return tensor, labels, labels_untouched

    def get_word_index(self, word):
        """
        Returns the index of a word if known, the one of <unk> otherwise
        """
        if word in self.index_to_words:
            return self.words_to_index[word]
        return self.words_to_index['<unk>']

    def bpe(self, token):
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        if token in self.cache:
            return self.cache[token]
        pairs = self.get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = self.get_pairs(word)
        word = ' '.join(word)
        if word == '\n  </w>':
            word = '\n</w>'
        self.cache[token] = word
        return word


    def get_tensor_Transformer(self, num_unknown):
        articles = [article['content'] for article in self.articles['fake']] + [article['content'] for article in
                                                                                self.articles['real']]
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
        for k in range(num_unknown):
            labels[k] = 0
        #articles, labels, labels_untouched = list(
        #    zip(*np.random.permutation(list(zip(articles, labels, labels_untouched)))))
        tensor = self.encode(articles)
        maximum_taille_article = max([len(i) for i in tensor])
        tensor_final = torch.zeros(512, maximum_taille_article).long()
        for index, val  in enumerate(tensor):
            for index_encode, encode in enumerate(val):
                tensor_final[index, index_encode] = encode
        embedding = self.model(tensor_final)[0][:len(labels)]

        return embedding, labels, labels_untouched

    def encode(self, texts):
        self.nlp = spacy.load('en', disable=['parser', 'tagger', 'ner', 'textcat'])
        texts_tokens = []
        for text in texts:
            text = ' '.join(text)
            text = self.nlp(self.text_standardize(ftfy.fix_text(text)))
            text_tokens = []
            for token in text:
                text_tokens.extend([self.encoder.get(t, 0) for t in self.bpe(token.text.lower()).split(' ')])
            texts_tokens.append(text_tokens)
        return texts_tokens


    @staticmethod
    def get_parafac_decomposition(tensor, rank):
        """
        Returns
        :param tensor:
        :param rank:
        :return: 3 matrix: (vocab, rank) (vocab, rank) and (num of articles, rank)
        """
        return tucker(tensor, rank=rank)


    def get_pairs(self, word):
        """
        Return set of symbol pairs in a word.
        word is represented as tuple of symbols (symbols being variable-length strings)
        """
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs


    def text_standardize(self, text):
        """
        fixes some issues the spacy tokenizer had on books corpus
        also does some whitespace standardization
        """
        text = text.replace('—', '-')
        text = text.replace('–', '-')
        text = text.replace('―', '-')
        text = text.replace('…', '...')
        text = text.replace('´', "'")
        text = re.sub(r'''(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)''', r' \1 ', text)
        text = re.sub(r'\s*\n\s*', ' \n ', text)
        text = re.sub(r'[^\S\n]+', ' ', text)
        return text.strip()
