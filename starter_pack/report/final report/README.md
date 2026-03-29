Math4AI Capstone Report

This repository contains a LaTeX report for a capstone project in the Math4AI program organized by the National AI Center – AI Academy.

Project Question

When does a one-hidden-layer neural network outperform a linear classifier, and when is additional complexity unnecessary?

Overview

The project compares two models implemented from scratch in NumPy:

Softmax Regression (linear classifier)
One-hidden-layer neural network with tanh activation

These models are evaluated on three datasets:

Linear Gaussian (linearly separable)
Moons (nonlinear structure)
Digits (real-world classification task)
Report Structure
Introduction

Defines the research question, hypotheses, and motivation for comparing linear and nonlinear models.

Background

Covers the mathematical foundations:

Linear algebra (SVD, conditioning)
PCA and dimensionality reduction
Information theory (entropy, KL divergence, cross-entropy)
Optimization methods (SGD, Momentum, Adam)
Probabilistic interpretation of classification
Methods

Describes:

Dataset generation and preprocessing
Model architectures
Training procedure and hyperparameters
Evaluation metrics
Reproducibility setup
Experiments

Includes:

Comparison on synthetic datasets
Digits benchmark evaluation
Capacity ablation (hidden layer size)
Optimizer comparison
Failure case analysis (under-capacity network)
Repeated-seed evaluation for statistical significance
Advanced Track

Optional analysis using PCA/SVD or prediction uncertainty.

Discussion

Interprets when linear models are sufficient and when nonlinear models provide advantages.

Limitations

Discusses dataset constraints, model simplicity, and generalization limits.
