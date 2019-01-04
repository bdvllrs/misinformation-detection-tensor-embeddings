Python implementation of 
_Guacho, G. B., Abdali, S., Shah, N., & Papalexakis, 
E. E. (2018). Semi-supervised Content-based Detection of 
Misinformation via Tensor Embeddings. arXiv preprint arXiv:1804.09088._

# Install guide

Requires python >= 3.6.

## Configuration
Copy the `config/config.default.json` and rename the copy `config/config.json`.

This file will contain the configuration of the project.
## To use sparse matrices
Uses last (not even released) tensorly sparse features.

Install master version of sparse:
```bash
$ git clone https://github.com/pydata/sparse/ & cd sparse
$ pip install .
```

Then use this version of tensorly:
```bash
$ git clone https://github.com/jcrist/tensorly.git tensorly-sparse & cd tensorly-sparse
$ git checkout sparse-take-2
```
Then place the `tensorly-sparse/tensorly` folder our project structure.
# Configuration file for different method

## Method for the embedding of the article:

There are 2 choices : Method of the co-occurence matrix or embedding with GloVe.

### Co-occurence matrix

`method_decomposition_embedding` can be `parafac` or `GloVe`.
```json
{
    "method_decomposition_embedding": "parafac",
    "rank_parafac_decomposition": 10,
    "size_word_co_occurrence_window": 5,
    "use_frequency": false
}
```

### GloVe

The embedding with glove : download GloVe nlp.stanford.edu/data/glove.6B.zip

There is 2 method of embedding : mean or RNN

```json
{
    "method_decomposition_embedding": "GloVe",
    "method_embedding_glove": "mean",
    "GloVe_adress": "../glove6B/glove.6B.100d.txt"
}
```

### Transformer

Git clone the project transformer-pytorch-hugging face, rename the file transformer and download the pre-trained model
of OpenAI. Set the config path :

```json
{
  "encoder_path":"transformer/model/encoder_bpe_40000.json",
  "bpe_path":"transformer/model/vocab_40000.bpe",
  "method_decomposition_embedding":"Transformer"
  }
```

### LDA-idf

```json
{
  "method_decomposition_embedding":"LDA"
  }
```

# WMD
Install packages:

- spacy
- wmd


# Credits to
- Kipf, Thomas N and Welling, Max, _Semi-Supervised Classification with Graph Convolutional Networks_. [https://github.com/tkipf/pygcn]().
- TensorLy: Tensor Learning in Python, Jean Kossaifi and Yannis Panagakis and Anima Anandkumar and Maja Pantic
- Vlad Niculae, Matt Kusner for the word mover's distance knn.

