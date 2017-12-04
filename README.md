# GaussDCA
Python (Cython) implementation of GaussDCA adapted from [here](https://github.com/carlobaldassi/GaussDCA.jl).

For the original paper please refer to ["Fast and accurate multivariate Gaussian modeling of protein families: Predicting residue contacts and protein-interaction partners"](doi:10.1371/journal.pone.0092721) by Carlo Baldassi, Marco Zamparo, Christoph Feinauer, Andrea Procaccini, Riccardo Zecchina, Martin Weigt and Andrea Pagnani, (2014) PLoS ONE 9(3): e92721. 

This version implements what is called the "slow fallback" in the original Julia implementation. 

## Installation
Runs in Python 3.6
1. Make sure [cython](http://docs.cython.org/en/latest/src/quickstart/install.html) and [numpy]() are installed and up to date: `pip install Cython` and `pip install numpy`.
2. Compile the cython source code: `cd src; python setup.py build_ext -i; cd ..`

## Usage
```python src/gaussdca.py [-h] [-o OUTPUT] [-s SEPARATION] [-t THREADS] alignment_file```

So far the alignment file needs to be in a3m format (with or without insertions). The output will be printed or saved into a file if given. Sequence separation and the number of threads for multiprocessing can be specified.

TODO: implement the faster version with alignment compression.
