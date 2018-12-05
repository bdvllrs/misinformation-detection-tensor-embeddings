from utils.ArticlesProvider import ArticlesProvider
from decomposition.Decomposition import Decomposition
from utils import Config
import numpy as np
import torch
import spacy
import ftfy
import re
import json
from transformer.model_pytorch import TransformerModel, load_openai_pretrained_model, DEFAULT_CONFIG

class TransformerDecomposition(Decomposition):

    def __init__(self, config: Config, articles: ArticlesProvider):
        super().__init__(config, articles)
        self.encoder = json.load(open(config["encoder_path"]))
        merges = open(config["bpe_path"], encoding='utf-8').read().split('\n')[1:-1]
        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.model = TransformerModel(DEFAULT_CONFIG)
        load_openai_pretrained_model(self.model, path='./transformer/model/', path_names='./transformer')
        self.cache = {}

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


    def apply(self):
        articles = [article['content'] for article in self.articles.articles['fake']] + [article['content'] for article in
                                                                                self.articles.articles['real']]

        tensor = self.encode(articles)
        maximum_taille_article = max([len(i) for i in tensor])
        tensor_final = torch.zeros(512, maximum_taille_article).long()
        for index, val  in enumerate(tensor):
            for index_encode, encode in enumerate(val):
                tensor_final[index, index_encode] = encode
        embedding = self.model(tensor_final)[0][:len(articles)]

        return embedding

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
