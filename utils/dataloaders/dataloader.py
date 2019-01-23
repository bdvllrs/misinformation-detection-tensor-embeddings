from utils import Config
import nltk


class DataLoader:
    def __init__(self, config: Config):
        self.config = config
        self.articles = {
            'fake': [],
            'real': []
        }
        self.original_articles = {'fake': [], 'real': []}
        self.vocabulary = {}
        self.frequency = {}  # dict : keys Words et values : list of files where the words are from

    def load(self):
        """
        Load data from list of folder
        :return:
        """
        raise NotImplementedError

    def _get_content(self, uid, content: str, type: str = 'fake'):
        """
        Get the content of a given file
        :param filename: path to file to open
        """
        ps = nltk.PorterStemmer()
        content = content.replace('\n', '').replace('\r', '').replace("\\'", "'")
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
                self.frequency[stemmed_word] = [uid]
            else:
                if uid not in self.frequency[stemmed_word]:
                    self.frequency[stemmed_word].append(uid)
        return content_words_tokenized
