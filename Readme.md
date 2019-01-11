### AGNN PyTorch Geometric

This branch contains implementation of [Attention-based Graph Neural Network for Semi-supervised Learning](https://arxiv.org/pdf/1803.03735.pdf) in [Pytorch Geometric](https://rusty1s.github.io/pytorch_geometric/build/html/index.html)

Note: The code has been tested on Ubuntu 18.04, PyTorch 1.0, CUDA-10.0, Python 3.6

To Install PyTorch Geometric Click [here](https://rusty1s.github.io/pytorch_geometric/build/html/notes/installation.html). Install PyTorch Geometric by building from Source and not using pip.

To run the code
* python agnn_citeseer.py
* python agnn_cora.py
* python agnn_pubmed.py

#### Results  
Classification Accuracy has been used as the performance metric

|  Datasets      |   Val        |   Test     |
| :------------- | :--------:   | -------: |
|   Pubmed       | 0.8060       |   0.7840  |  
|   Cora         | 0.80         |   0.7910  |
|   Citeseer     | 0.69         | 0.6820    |

