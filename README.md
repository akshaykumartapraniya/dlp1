# Randomized Nonnegative Matrix Factorization

Reproduction of Erichson, Mendible, Wihlborn, Kutz (2018),
*Randomized Nonnegative Matrix Factorization*, Pattern Recognition Letters 104, 1-7.

## Contents
- `randomized_nmf_implementation.ipynb` - CPU implementation (NumPy): probabilistic
  compression, deterministic HALS, randomized HALS, compressed MU baseline.
- `randomized_nmf_gpu.ipynb` - CUDA implementation (PyTorch), including a
  streaming compressor for out-of-core factorization.
- `presentation/` - Beamer slides summarizing the results.

## Requirements
NumPy, SciPy, Matplotlib, scikit-learn, Jupyter.
GPU notebook additionally needs PyTorch built for your CUDA version.
