# General :

Our work proposes a graph-based semi-supervised fake news detection method.


First, this repo is a python implementation of the paper

*Semi-Supervised Learning and Graph Neural Networks for Fake News Detection* by Adrien Benamira, Benjamin Devillers,Etienne Lesot,
Ayush K., Manal Saadi and Fragkiskos D. Malliaros

published at ASONAM '19, August 27-30, 2019, Vancouver, Canada
copyright space2019Association for Computing Machinery. 
ACM ISBN 978-1-4503-6868-1/19/08
http://dx.doi.org/10.1145/3341161.3342958}



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

`method_decomposition_embedding` can be `parafac`, `GloVe`, `LDA` or `Transformer`.
```yaml
embedding:
  # Parafac - LDA - GloVe - Transformer -
  method_decomposition_embedding: parafac
  method_embedding_glove: mean 
  rank_parafac_decomposition: 10
  size_word_co_occurrence_window: 5
  use_frequency: No  # If No, only a binary co-occurence matrix.
  vocab_size: -1
```

### GloVe

The embedding with glove : download GloVe nlp.stanford.edu/data/glove.6B.zip

There is 2 method of embedding: mean or RNN

```yaml
paths:
  GloVe_adress: ../glove6B/glove.6B.100d.txt

embedding:
  # Parafac - LDA - GloVe - Transformer -
  method_decomposition_embedding: GloVe
  method_embedding_glove: mean  # mean or RNN
  use_frequency: No
  vocab_size: -1
```

### Transformer

Git clone the project transformer-pytorch-hugging face, rename the file transformer and download the pre-trained model
of OpenAI. Set the config path :

```yaml
paths:
  encoder_path: transformer/model/encoder_bpe_40000.json
  bpe_path: transformer/model/vocab_40000.bpe

embedding:
  # Parafac - LDA - GloVe - Transformer -
  method_decomposition_embedding: Transformer
  use_frequency: No
  vocab_size: -1
```

### LDA-idf

```yaml
embedding:
  # Parafac - LDA - GloVe - Transformer -
  method_decomposition_embedding: LDA
  use_frequency: No
  vocab_size: -1
```

## Word Mover Distance

The idea is instead of using the euclidean distance, we can use the [WMD](http://proceedings.mlr.press/v37/kusnerb15.pdf)


Install Python packages:

- spacy
- wmd

## Learning Methods

### PyGCN
The `pygcn` lib used is: [tkipf pytorch implementation](https://github.com/tkipf/pygcn).

### PyAGNN
The `pyagnn` lib used is based on [dawnrange pytorch implementation](https://github.com/dawnranger/pytorch-AGNN).

# Results

Our pipeline is described in [our report](https://github.com/bdvllrs/misinformation-detection-tensor-embeddings/blob/Asonam/fig_and_report/ASONAM_2019_paper_285(2).pdf)

Here is our result on the dataset

![Alt text](fig_and_report/fig_compare_algo.png?raw=true "Title")
![Alt text](fig_and_report/fig_compare_embedding_agnn.png?raw=true "Title")
![Alt text](fig_and_report/fig_compare_voisin_agnn.png?raw=true "Title")


# Credits to
- Kipf, Thomas N and Welling, Max, _Semi-Supervised Classification with Graph Convolutional Networks_. [https://github.com/tkipf/pygcn]().
- TensorLy: Tensor Learning in Python, Jean Kossaifi and Yannis Panagakis and Anima Anandkumar and Maja Pantic
- Vlad Niculae, Matt Kusner for the word mover's distance knn.
- Attention-based Graph Neural Network for semi-supervised learning,

