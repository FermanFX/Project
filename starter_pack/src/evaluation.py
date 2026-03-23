import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

from .models import SoftmaxRegression, OneHiddenLayerNN


@dataclass
class ExperimentResult:
    """Container for experiment results."""
    model_name: str
    dataset_name: str
    test_accuracy: float
    test_loss: float
    test_confidence: Optional[float] = None
    test_entropy: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'model': self.model_name,
            'dataset': self.dataset_name,
            'test_accuracy': float(self.test_accuracy),
            'test_loss': float(self.test_loss),
        }


@dataclass
class RepeatedSeedResult:
    """
    Results from repeated seed evaluation.
    
    Required for digits benchmark:
    - Run with 5 different seeds
    - Report mean ± 95% CI
    - CI formula: mean ± 2.776 * std / sqrt(5)
    """
    model_name: str
    accuracies: List[float]
    losses: List[float]
    
    @property
    def mean_accuracy(self) -> float:
        return np.mean(self.accuracies)
    
    @property
    def std_accuracy(self) -> float:
        # TODO: ddof=1 for sample standard deviation
        return np.std(self.accuracies, ddof=1)
    
    @property
    def mean_loss(self) -> float:
        return np.mean(self.losses)
    
    @property
    def std_loss(self) -> float:
        return np.std(self.losses, ddof=1)
    
    @property
    def ci_95_accuracy(self) -> Tuple[float, float]:
        """
        95% confidence interval for accuracy mean.
        
        Using t-distribution with n-1 degrees of freedom:
            CI = mean ± t_{0.975, n-1} * std / sqrt(n)
        
        For n=5: t_{0.975, 4} = 2.776
        """
        # TODO: n = len(self.accuracies)
        # TODO: t_crit = 2.776 (for n=5, 95% CI)
        # TODO: se = self.std_accuracy / np.sqrt(n)
        # TODO: return (self.mean_accuracy - t_crit * se, self.mean_accuracy + t_crit * se)
        pass
    
    @property
    def ci_95_loss(self) -> Tuple[float, float]:
        # TODO: Same as ci_95_accuracy but for losses
        pass
    
    def summary(self) -> str:
        return (
            f"{self.model_name}:\n"
            f"  Accuracy: {self.mean_accuracy:.4f} ± {self.std_accuracy:.4f} "
            f"95% CI [{self.ci_95_accuracy[0]:.4f}, {self.ci_95_accuracy[1]:.4f}]\n"
            f"  Loss:     {self.mean_loss:.4f} ± {self.std_loss:.4f} "
            f"95% CI [{self.ci_95_loss[0]:.4f}, {self.ci_95_loss[1]:.4f}]"
        )


class Evaluator:
    """
    Evaluation utilities for classification models.
    """
    
    def __init__(self):
        pass
    
    def predict_proba(self, model, X: np.ndarray) -> np.ndarray:
        """
        Get predicted probabilities.
        
        Args:
            model: Trained model (SoftmaxRegression or OneHiddenLayerNN)
            X: Input features
        
        Returns:
            Probability matrix (n_samples, n_classes)
        """
        # TODO: if isinstance(model, SoftmaxRegression):
        # TODO:     _, P = model.forward(X)
        # TODO: else:
        # TODO:     cache = model.forward(X)
        # TODO:     P = cache['P']
        # TODO: return P
        pass
    
    def predict(self, model, X: np.ndarray) -> np.ndarray:
        """
        Get predicted class labels.
        
        Args:
            model: Trained model
            X: Input features
        
        Returns:
            Predicted labels (n_samples,)
        """
        # TODO: P = self.predict_proba(model, X)
        # TODO: return np.argmax(P, axis=1)
        pass
    
    def compute_metrics(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Returns:
            Dictionary with:
            - accuracy: fraction of correct predictions
            - cross_entropy: mean negative log-probability
            - confidence: mean of max probabilities
            - entropy: mean predictive entropy
        """
        # TODO: P = self.predict_proba(model, X)
        # TODO: y_pred = np.argmax(P, axis=1)
        
        # TODO: accuracy = np.mean(y_pred == y)
        
        # TODO: n = len(y)
        # TODO: eps = 1e-9
        # TODO: cross_ent = -np.mean(np.log(P[np.arange(n), y] + eps))
        
        # TODO: confidence = np.mean(np.max(P, axis=1))
        
        # TODO: entropy = -np.mean(np.sum(P * np.log(P + eps), axis=1))
        
        # TODO: return {...}
        pass
    
    def confidence_by_bin(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        n_bins: int = 5
    ) -> List[Dict]:
        """
        Compute confidence vs accuracy by confidence bins.
        
        For Track B: Prediction confidence and reliability.
        
        Confidence = max predicted probability for each sample.
        
        Algorithm:
            1. Get max probability for each sample
            2. Bin samples by confidence level
            3. Compute accuracy within each bin
        
        Args:
            model: Trained model
            X: Features
            y: True labels
            n_bins: Number of confidence bins
        
        Returns:
            List of dicts:
            [{'bin': 1, 'conf_range': (0, 0.2), 'mean_confidence': 0.1, 
              'accuracy': 0.15, 'count': 50}, ...]
        """
        # TODO: P = self.predict_proba(model, X)
        # TODO: max_probs = np.max(P, axis=1)
        # TODO: y_pred = np.argmax(P, axis=1)
        # TODO: correct = (y_pred == y)
        
        # TODO: bin_edges = np.linspace(0, 1, n_bins + 1)
        
        # TODO: results = []
        # TODO: for i in range(n_bins):
        # TODO:     mask = (max_probs >= bin_edges[i]) & (max_probs < bin_edges[i+1])
        # TODO:     if np.sum(mask) > 0:
        # TODO:         results.append({...})
        # TODO: return results
        pass
    
    def repeated_seed_evaluation(
        self,
        model_class,
        model_kwargs: Dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        seeds: List[int],
        epochs: int = 200,
        batch_size: int = 64,
        optimizer_class=None,
        optimizer_kwargs: Dict = None
    ) -> RepeatedSeedResult:
        """
        Run training with multiple random seeds.
        
        REQUIRED for digits benchmark final reporting.
        
        Protocol:
            1. For each seed:
               - Set random seed
               - Initialize model
               - Train on X_train, y_train
               - Use X_val, y_val for checkpoint selection
               - Evaluate on X_test, y_test
            2. Return RepeatedSeedResult with all seeds
        
        Args:
            model_class: SoftmaxRegression or OneHiddenLayerNN
            model_kwargs: Keyword arguments for model
            seeds: List of 5 random seeds
            ... (other training args)
        
        Returns:
            RepeatedSeedResult with accuracies and losses
        """
        accuracies = []
        losses = []
        
        for seed in seeds:
            pass
            # # TODO: np.random.seed(seed)
            # 
            # # TODO: model = model_class(**model_kwargs)
            # 
            # # TODO: Create trainer with appropriate optimizer
            # 
            # # TODO: trainer.train(X_train, y_train, X_val, y_val)
            # 
            # # TODO: metrics = self.compute_metrics(model, X_test, y_test)
            # 
            # # TODO: accuracies.append(metrics['accuracy'])
            # # TODO: losses.append(metrics['cross_entropy'])
        
        return RepeatedSeedResult(
            model_name=model_kwargs.get('num_classes', 'NN'),
            accuracies=accuracies,
            losses=losses
        )


def gradient_check(
    model,
    X: np.ndarray,
    Y: np.ndarray,
    epsilon: float = 1e-5,
    verbose: bool = True
) -> bool:
    """
    Verify gradients using finite differences.
    
    REQUIRED sanity check for implementation verification.
    
    Method:
        1. Compute numerical gradient using finite differences:
           ∂L/∂θ ≈ (L(θ + ε) - L(θ - ε)) / (2ε)
        
        2. Compare with analytical gradient from backprop
        
        3. Check relative error < 1e-5
    
    Args:
        model: Model to check
        X: Small batch of inputs (use 10-100 samples)
        Y: One-hot labels
        epsilon: Finite difference step
        verbose: Print results
    
    Returns:
        True if gradients are correct (relative error < 1e-5)
    """
    # TODO: cache = model.forward(X)
    # TODO: grads = model.backward(X, Y, cache['P'] if dict else cache[1])
    
    # TODO: for each parameter:
    # TODO:     for each element:
    # TODO:         param[i,j] += epsilon
    # TODO:         loss_high = forward_pass_loss(...)
    # TODO:         param[i,j] -= 2*epsilon
    # TODO:         loss_low = forward_pass_loss(...)
    # TODO:         param[i,j] += epsilon
    # TODO:         
    # TODO:         num_grad = (loss_high - loss_low) / (2*epsilon)
    # TODO:         ana_grad = analytical_grad[i,j]
    # TODO:         rel_error = |num - ana| / (|num| + |ana| + eps)
    
    # TODO: return True if all rel_error < 1e-5
    pass


def check_probability_sum(P: np.ndarray, eps: float = 1e-6) -> bool:
    """
    Verify that predicted probabilities sum to 1.
    
    REQUIRED sanity check: softmax output must be normalized.
    
    Args:
        P: Probability matrix (n_samples, n_classes)
        eps: Tolerance
    
    Returns:
        True if all rows sum to 1 (within tolerance)
    """
    # TODO: sums = np.sum(P, axis=1)
    # TODO: all_one = np.allclose(sums, 1.0, atol=eps)
    # TODO: return all_one
    pass


def check_nan_inf(model, X: np.ndarray) -> bool:
    """
    Check for NaN or Inf in model outputs.
    
    REQUIRED sanity check: prevents silent failures.
    
    NaN/Inf can occur from:
        - Division by zero
        - Log of zero
        - Overflow in exp()
        - Gradient explosion
    
    Args:
        model: Model to check
        X: Input data
    
    Returns:
        True if no NaN/Inf found (model is healthy)
    """
    # TODO: cache = model.forward(X)
    
    # TODO: if dict:
    # TODO:     for key, val in cache.items():
    # TODO:         check np.isnan(val) and np.isinf(val)
    # TODO: else:
    # TODO:     for val in cache:
    # TODO:         check np.isnan and np.isinf
    
    # TODO: return True if healthy
    pass
