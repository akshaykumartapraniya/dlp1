# Randomized Nonnegative Matrix Factorization

Reproduction of Erichson, Mendible, Wihlborn, Kutz (2018),
*Randomized Nonnegative Matrix Factorization*, Pattern Recognition Letters 104, 1-7.

## Contents
- `nmf.ipynb` - CUDA implementation (PyTorch), including a
  streaming compressor for out-of-core factorization.

## Requirements
NumPy, SciPy, Matplotlib, scikit-learn, Jupyter.
GPU notebook additionally needs PyTorch built for CUDA version.
