# Math4AI Capstone Report

This repository contains a LaTeX report for a capstone project in the Math4AI program organized by the National AI Center – AI Academy.

---

## Project Question

When does a one-hidden-layer neural network outperform a linear classifier, and when is additional complexity unnecessary?

---

## Overview

We compare two models implemented from scratch in NumPy:

- Softmax Regression (linear classifier)
- One-hidden-layer Neural Network (tanh activation)

These models are evaluated on:

- Linear Gaussian (linearly separable)
- Moons (nonlinear structure)
- Digits (real-world classification)

---

## Report Structure

### Introduction
Defines the research question, hypotheses, and motivation for comparing linear and nonlinear models.

### Background
Covers mathematical foundations:
- Linear algebra (SVD, conditioning, PCA)
- Information theory (entropy, KL divergence, cross-entropy)
- Optimization (SGD, Momentum, Adam)
- Probabilistic interpretation of classification

### Methods
Includes:
- Dataset generation and preprocessing
- Model architectures (Softmax vs NN)
- Training protocol and hyperparameters
- Evaluation metrics
- Reproducibility setup

### Experiments
Includes:
- Synthetic dataset comparison
- Digits benchmark results
- Capacity ablation (hidden size study)
- Optimizer comparison
- Failure case (under-capacity network)
- Repeated-seed evaluation

### Advanced Track
Optional PCA/SVD analysis or uncertainty estimation.

### Discussion
Explains when linear models are sufficient and when nonlinear models are needed.

### Limitations
Discusses:
- Limited dataset variety
- Single hidden-layer architecture
- Fixed hyperparameters
- Limited statistical testing for all experiments

---

## Requirements

To compile the LaTeX report:

- TeX Live or MiKTeX
- Packages:
  - geometry
  - amsmath
  - amssymb
  - graphicx
  - booktabs
  - hyperref
  - enumitem
  - tabularx
  - microtype

---

## Compilation

```bash
pdflatex main.tex
