# GaussDCA
Python (Cython) implementation of GaussDCA adapted from [here](https://github.com/carlobaldassi/GaussDCA.jl).

For the original paper please refer to ["Fast and accurate multivariate Gaussian modeling of protein families: Predicting residue contacts and protein-interaction partners"](doi:10.1371/journal.pone.0092721) by Carlo Baldassi, Marco Zamparo, Christoph Feinauer, Andrea Procaccini, Riccardo Zecchina, Martin Weigt and Andrea Pagnani, (2014) PLoS ONE 9(3): e92721. 

This version implements what is called the "slow fallback" in the original Julia implementation. 

TODO: implement the faster version with alignment compression.
