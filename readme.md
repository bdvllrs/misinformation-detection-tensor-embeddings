# General :

Our work proposes a graph-based semi-supervised fake news detection method.


First, this repo is a python implementation of
_Guacho, G. B., Abdali, S., Shah, N., & Papalexakis, 
E. E. (2018). Semi-supervised Content-based Detection of 
Misinformation via Tensor Embeddings. arXiv preprint arXiv:1804.09088._


Then we proposed to compare the leanring method based on FaBP to a graph neural method one.



# Install guide

Requires python >= 3.6.

## Configuration
Copy the `config/config.default.yaml` and rename the copy `config/config.yaml`.

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

## To use Transformer

Install master version of sparse:
```bash
$ git clone https://github.com/huggingface/pytorch-openai-transformer-lm.git
```
 Then place the `pytorch-openai-transformer-lm/` folder in our project structure under teh name `transformer`


# Methods

## Method for article embedding and construct the graph:

There are multiple choices : Method of the co-occurence matrix / embedding with GloVe (mean or RNN) / Transformer /
LDA-idf.

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

## Word Mover Distance

The idea is instead of using the euclidean distance, we can use the [WMD](http://proceedings.mlr.press/v37/kusnerb15.pdf)


Install packages:

- spacy
- wmd

## Learning Methods

### PyGCN
The `pygcn` lib used is: [tkipf pytorch implementation](https://github.com/tkipf/pygcn).

### PyAGNN
The `pyagnn` lib used is based on [dawnrange pytorch implementation](https://github.com/dawnranger/pytorch-AGNN).

# Results

Our pipeline is described in [our report](XXX)

Here is our result on the dataset

![Alt text](fig_and_report/fig_compare_algo.png?raw=true "Title")
![Alt text](fig_and_report/fig_compare_embedding_agnn.png?raw=true "Title")
![Alt text](fig_and_report/fig_compare_voisin_agnn.png?raw=true "Title")


# Credits to
- Kipf, Thomas N and Welling, Max, _Semi-Supervised Classification with Graph Convolutional Networks_. [https://github.com/tkipf/pygcn]().
- TensorLy: Tensor Learning in Python, Jean Kossaifi and Yannis Panagakis and Anima Anandkumar and Maja Pantic
- Vlad Niculae, Matt Kusner for the word mover's distance knn.
- Attention-based Graph Neural Network for semi-supervised learning,

