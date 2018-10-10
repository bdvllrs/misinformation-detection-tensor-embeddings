import os
import nltk
import numpy as np


def get_fullpath(*path):
    """
    Returns an absolute path given a relative path
    """
    path = [os.path.curdir] + list(path)
    return os.path.abspath(os.path.join(*path))

def loadGloveModel(gloveFile):
    """
    :param gloveFile: adress of glove file
    :return:
    """
    print("Loading Glove Model")
    f = open(gloveFile, 'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model


class ArticleTensor:
    def __init__(self, path):
        """
        :param path: path to all the articles. Must contain Fake, Fake_title, Real, Real_title
        """
        self.glove=loadGloveModel('../glove6B/glove.6B.100d.txt')
        self.path = path
        self.nbre_all_article=0
        self.vocabulary = {}
        self.frequency={}       #dictinnaire : clefs Words et attributs : liste de files dans lesquels ces mots sont
        self.index_to_words = []
        self.words_to_index = {}
        self.articles = {
            'fake': [],
            'real': []
        }

    def get_content(self, filename):
        """
        Get the content of a given file
        :param filename: path to file to open
        """
        with open(filename, 'r', encoding="utf-8", errors='ignore') as document:
            content = document.read().replace('\n', '').replace('\r', '')
        content_words_tokenized = nltk.word_tokenize(content.lower())
        # Add words in the vocab
        for word in content_words_tokenized:
            self.vocabulary[word] = 1 if word not in self.vocabulary.keys() else self.vocabulary[word] + 1
            #si 1 mots a un attribut de la taille du dataset c'est qu'il est dans tous les articles. On le supprime du voc
            if word not in self.frequency.keys():
                self.frequency[word] =[filename]
            else:
                if filename not in self.frequency[word]:
                    self.frequency[word].append(filename)
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

    def get_glove_matrix(self, article,method="mean"):
        """
        Get the Glove_mean of an article
        :param article
        """
        vector = np.zeros(100)
        N=0
        for k, word in enumerate(article):
            if word in self.vocabulary and len(self.frequency[word])<self.nbre_all_article:
                if method=="mean":
                    N+=1
                    try:
                        vector=vector+self.glove[word]
                    except Exception:
                        vector = vector + self.glove['unk']
                if method=="BoW":
                    pass
        print("Nombre de mots considéré", N)
        return vector/N

    def get_tensor(self, num_unknown):
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
        articles, labels, labels_untouched = list(zip(*np.random.permutation(list(zip(articles, labels, labels_untouched)))))
        tensor = np.zeros((100, len(articles)))
        for k, article in enumerate(articles):
            tensor[:, k] = self.get_glove_matrix(article)
        return tensor, labels, labels_untouched

    def get_word_index(self, word):
        """
        Returns the index of a word if known, the one of <unk> otherwise
        """
        if word in self.index_to_words:
            return self.words_to_index[word]
        return self.words_to_index['<unk>']



