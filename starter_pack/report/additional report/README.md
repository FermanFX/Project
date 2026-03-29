# Math4AI Capstone – Extended Report

This repository contains the **extended version** of the Math4AI capstone project report.

It provides a deeper and more technical exploration of the main project results and theoretical foundations.

---

## 📄 Overview

The extended report expands on the main study:

> When does a one-hidden-layer neural network outperform a linear classifier, and when is additional complexity unnecessary?

While the main report focuses on results and interpretation, this version includes detailed derivations, deeper explanations, and additional experimental insights.

---

## 📚 What is Included

### 🧠 Mathematical Foundations
- Linear algebra (SVD, matrix conditioning, projections)
- Principal Component Analysis (PCA)
- Information theory (entropy, KL divergence, cross-entropy)
- Probability theory for classification

### ⚙️ Optimization Theory
- Gradient Descent derivation and intuition
- Stochastic Gradient Descent (SGD)
- Momentum-based optimization
- Adam optimizer behavior and comparison

### 🧪 Model Analysis
- Softmax regression as a linear decision rule
- One-hidden-layer neural network expressiveness
- Role of non-linearity (tanh activation)
- Decision boundary interpretation

### 📊 Experimental Deep Dive
- Capacity vs performance analysis
- Hidden layer size ablation study
- Failure cases of under-capacity models
- Repeated-seed evaluation and variance analysis

### 🔍 Statistical Perspective
- Variance across random seeds
- Stability of training outcomes
- Confidence intervals for performance metrics

---

## 📈 Datasets Used

- **Linear Gaussian dataset**: linearly separable structure
- **Moons dataset**: nonlinear classification problem
- **Digits dataset**: real-world handwritten digit classification

---

## 🧩 Relationship to Main Report

- The **main report** presents summarized results and conclusions.
- The **extended report** provides:
  - Full mathematical explanations
  - Detailed derivations
  - Expanded experimental discussion
  - Deeper interpretation of results

---

## ⚙️ Requirements

To compile the LaTeX document:

- TeX Live or MiKTeX
- Required packages:
  - geometry
  - amsmath, amssymb
  - graphicx
  - booktabs
  - hyperref
  - enumitem
  - tabularx
  - microtype

---

## ▶️ Compilation

```bash
pdflatex extended.tex
pdflatex extended.tex
