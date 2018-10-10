# Sparse tensors
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

Python implementation of 
_Guacho, G. B., Abdali, S., Shah, N., & Papalexakis, 
E. E. (2018). Semi-supervised Content-based Detection of 
Misinformation via Tensor Embeddings. arXiv preprint arXiv:1804.09088._

## Credits to
- TensorLy: Tensor Learning in Python, Jean Kossaifi and Yannis Panagakis and Anima Anandkumar and Maja Pantic
