"""
PART 2.2: TREE-BASED MODELS (10 points)
========================================
Decision Tree, Random Forest, XGBoost - ALGORITHM parts marked with TODO
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


class DecisionTreeModel:
    """
    Decision Tree Classifier implementation.
    
    ALGORITHM:
    TODO: Implement decision tree using recursive splitting
    
    Key concepts:
    1. Gini Impurity (splitting criterion):
       TODO: Gini = 1 - sum(p_i^2) for all classes i
    
    2. Information Gain:
       TODO: IG = Gini(parent) - weighted_avg * Gini(children)
    
    3. Best split selection:
       TODO: For each feature, try all split points
             Select split with highest Information Gain
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
        
    def gini impurity(self, y):
        """
        Calculate Gini impurity for a node.
        
        ALGORITHM:
        TODO: Gini = 1 - sum((count_class_i / n)^2)
        """
        # TODO: Implement gini impurity calculation
        pass
    
    def information_gain(self, y, y_left, y_right):
        """
        Calculate information gain from a split.
        
        ALGORITHM:
        TODO: IG = Gini(parent) - (n_left/n * Gini(left) + n_right/n * Gini(right))
        """
        # TODO: Implement information gain calculation
        pass
    
    def find_best_split(self, X, y):
        """
        Find the best split point.
        
        ALGORITHM:
        TODO: 
        1. For each feature:
           - For each unique value as potential split point:
             - Split data into left/right
             - Calculate information gain
        2. Return best feature, best threshold, best gain
        """
        # TODO: Implement best split search
        pass
    
    def build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.
        
        ALGORITHM:
        TODO:
        1. Check stopping criteria (depth, min_samples, pure node)
        2. Find best split
        3. Recursively build left and right subtrees
        4. Return node with split info or leaf prediction
        """
        # TODO: Implement tree building
        pass
    
    def fit(self, X, y):
        """Train the decision tree."""
        self.tree = self.build_tree(np.array(X), np.array(y))
        
    def predict_single(self, x, node):
        """
        Predict single sample.
        
        ALGORITHM:
        TODO: Traverse tree until leaf node, return majority class
        """
        # TODO: Implement single prediction
        pass
    
    def predict(self, X):
        """Predict all samples."""
        return np.array([self.predict_single(x, self.tree) for x in X])


def train_sklearn_decision_tree(X_train, y_train, max_depth=5, random_state=42):
    """Train sklearn DecisionTreeClassifier."""
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state,
        criterion='gini'
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10, random_state=42):
    """
    Random Forest Classifier.
    
    ALGORITHM:
    TODO: Implement bagging with decision trees:
    1. Bootstrap sampling: Draw n_samples with replacement
    2. Train decision tree on each bootstrap sample
    3. For each split, only consider random subset of features
    4. Aggregate predictions (majority vote)
    
    Key parameters:
    - n_estimators: number of trees
    - max_features: number of features to consider at each split
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


class RandomForestModel:
    """
    Custom Random Forest implementation.
    
    ALGORITHM:
    TODO: Implement the following:
    
    1. Bootstrap sampling:
       TODO: Sample n examples with replacement from training data
    
    2. Train multiple decision trees on bootstrap samples
       TODO: Each tree sees different subset of data
    
    3. Feature randomness:
       TODO: At each split, consider only sqrt(n_features) random features
    
    4. Prediction (majority voting):
       TODO: Collect predictions from all trees
             Return most common class
    """
    
    def __init__(self, n_estimators=100, max_depth=10, max_features='sqrt', random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        
    def bootstrap_sample(self, X, y):
        """
        Create bootstrap sample.
        
        ALGORITHM:
        TODO: Randomly select n_samples indices with replacement
        """
        # TODO: Implement bootstrap sampling
        pass
    
    def fit(self, X, y):
        """
        Train all trees.
        
        ALGORITHM:
        TODO:
        1. For each estimator:
           - Create bootstrap sample
           - Train decision tree on bootstrap sample
        """
        # TODO: Implement fit method
        pass
    
    def predict(self, X):
        """
        Predict using majority voting.
        
        ALGORITHM:
        TODO:
        1. Get predictions from all trees
        2. For each sample, count votes
        3. Return majority class
        """
        # TODO: Implement prediction
        pass


def train_xgboost(X_train, y_train, n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42):
    """
    XGBoost Classifier.
    
    ALGORITHM:
    TODO: Implement gradient boosting:
    
    1. Initialize with base predictions (log-odds for classes)
    
    2. For each iteration (tree):
       - Calculate pseudo-residuals (gradient of loss)
       - Fit decision tree to pseudo-residuals
       - Update predictions using learning rate
    
    3. Loss function (Multi-class):
       TODO: L = sum(-sum(y_i * log(p_k)))
    
    4. Regularization:
       TODO: Add L1/L2 penalties to prevent overfitting
    
    Key equation for leaf weights:
    TODO: w = -sum(gradient) / (sum(hessian) + lambda)
    """
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train)
    return model


def compare_tree_models(X_train, y_train, X_test, y_test):
    """
    Compare all tree-based models.
    """
    print("=" * 60)
    print("TREE-BASED MODELS COMPARISON")
    print("=" * 60)
    
    results = {}
    
    dt = train_sklearn_decision_tree(X_train, y_train)
    results['Decision Tree'] = dt.score(X_test, y_test)
    
    rf = train_random_forest(X_train, y_train)
    results['Random Forest'] = rf.score(X_test, y_test)
    
    xgb = train_xgboost(X_train, y_train)
    results['XGBoost'] = xgb.score(X_test, y_test)
    
    for name, acc in results.items():
        print(f"{name}: {acc:.4f}")
    
    return results
