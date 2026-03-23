"""
PART 2.1: REGRESSION & CLASSIFICATION (15 points)
=================================================
Logistic Regression implementation - ALGORITHM parts marked with TODO
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def prepare_data(df, feature_cols, target_col='outcome', test_size=0.2, random_state=42):
    """
    Prepare data for modeling.
    Split into train/test and scale features.
    """
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


class LogisticRegressionModel:
    """
    Logistic Regression for multiclass classification.
    
    ALGORITHM:
    TODO: Implement multiclass logistic regression using gradient descent
    
    The logistic regression hypothesis for class k:
    TODO: h(x) = softmax(Wx + b) where:
          - W: weight matrix
          - b: bias vector
          - softmax(z_k) = exp(z_k) / sum(exp(z_j))
    
    Loss function (Cross-Entropy):
    TODO: J(W) = -sum(y * log(h(x))) / n
    
    Gradient:
    TODO: dJ/dW = X^T * (h(x) - y) / n
    
    Use gradient descent to optimize:
    TODO: W = W - learning_rate * dJ/dW
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, n_classes=3):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.n_classes = n_classes
        self.weights = None
        self.bias = None
        
    def softmax(self, z):
        """
        Softmax activation function.
        
        ALGORITHM:
        TODO: softmax(z_k) = exp(z_k) / sum(exp(z_j)) for all classes j
        
        Use numerically stable implementation:
        TODO: subtract max(z) from all elements before exp
        """
        # TODO: Implement softmax
        pass
    
    def fit(self, X, y):
        """
        Train the model using gradient descent.
        
        ALGORITHM:
        TODO: Implement the training loop:
        1. Initialize weights (n_features, n_classes) to zeros
        2. For each iteration:
           - Compute predictions using softmax
           - Calculate gradient
           - Update weights
        """
        # TODO: Implement fit method
        pass
    
    def predict(self, X):
        """
        Predict class labels.
        
        ALGORITHM:
        TODO: Return argmax of softmax outputs
        """
        # TODO: Implement predict method
        pass
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        ALGORITHM:
        TODO: Return softmax outputs
        """
        # TODO: Implement predict_proba method
        pass


def train_sklearn_logistic_regression(X_train, y_train, C=1.0, max_iter=1000):
    """
    Train sklearn LogisticRegression model.
    """
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        multi_class='multinomial',
        solver='lbfgs',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_custom_logistic_regression(X_train, y_train, learning_rate=0.01, n_iterations=1000):
    """
    Train custom LogisticRegression model.
    """
    model = LogisticRegressionModel(
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        n_classes=len(np.unique(y_train))
    )
    model.fit(X_train, y_train)
    return model


def compare_models(X_train, y_train, X_test, y_test):
    """
    Compare custom vs sklearn logistic regression.
    """
    print("=" * 60)
    print("LOGISTIC REGRESSION COMPARISON")
    print("=" * 60)
    
    sklearn_model = train_sklearn_logistic_regression(X_train, y_train)
    sklearn_acc = sklearn_model.score(X_test, y_test)
    print(f"\nSklearn Logistic Regression Accuracy: {sklearn_acc:.4f}")
    
    custom_model = train_custom_logistic_regression(X_train, y_train)
    custom_pred = custom_model.predict(X_test)
    custom_acc = np.mean(custom_pred == y_test)
    print(f"Custom Logistic Regression Accuracy: {custom_acc:.4f}")
    
    return sklearn_model, custom_model
