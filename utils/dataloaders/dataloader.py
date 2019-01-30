from utils import Config
import nltk


class DataLoader:
    def __init__(self, config: Config):
        self.config = config
        self.articles = {}
        self.original_articles = {}
        self.vocabulary = {}
        self.frequency = {}  # dict : keys Words et values : list of files where the words are from

    def load(self):
        """
        Load data from list of folder
        :return:
        """
        raise NotImplementedError

    def _get_content(self, uid, content: str, label, split_sentences=True):
        """
        Get the content of a given file
        :param filename: path to file to open
        """
        ps = nltk.PorterStemmer()
        content = content.replace('\n', '').replace('\r', '').replace("\\'", "'")
        if label not in self.original_articles.keys():
            self.original_articles[label] = []
        self.original_articles[label].append(content)
        content = content.split('.')  # Split sentences
        content_words_tokenized = []
        for sentence in content:
            content_words_tokenized.append(nltk.word_tokenize(sentence.lower()))
        # Add words in the vocab

        final_content = []

        for k, sentence in enumerate(content_words_tokenized):
            stemmed_words = []
            if len(sentence) > 1:
                for word in sentence:
                    # stemmed_word = ps.stem(word)
                    stemmed_word = word
                    # stemmed_word = word
                    self.vocabulary[stemmed_word] = (1 if stemmed_word not in self.vocabulary.keys()
                                                     else self.vocabulary[stemmed_word] + 1)

                    if stemmed_word not in self.frequency.keys():
                        self.frequency[stemmed_word] = [uid]
                    else:
                        if uid not in self.frequency[stemmed_word]:
                            self.frequency[stemmed_word].append(uid)
                    stemmed_words.append(stemmed_word)

            if len(stemmed_words) > 0:
                final_content.append(stemmed_words)

        if not split_sentences or not self.config.graph.sentence_based:  # If not sentence based, flatten the array
            final_content = [word for sentence in final_content for word in sentence]

        return final_content
