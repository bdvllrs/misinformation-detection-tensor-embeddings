import nltk
ps = nltk.PorterStemmer()

sentence = "gaming, the gamers play games"
words = nltk.word_tokenize(sentence)
list = []
for word in words:
    list.append(ps.stem(word))
print(list)