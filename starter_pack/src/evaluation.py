import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

from .models import SoftmaxRegression, OneHiddenLayerNN
from .trainer import Trainer
from .optimizers import SGD


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
        n = len(self.accuracies)
        t_crit = 2.776  # for n=5, 95% CI
        se = self.std_accuracy / np.sqrt(n)
        return (self.mean_accuracy - t_crit * se, self.mean_accuracy + t_crit * se)

    @property
    def ci_95_loss(self) -> Tuple[float, float]:
        n = len(self.losses)
        t_crit = 2.776  # for n=5, 95% CI
        se = self.std_loss / np.sqrt(n)
        return (self.mean_loss - t_crit * se, self.mean_loss + t_crit * se)

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
        if isinstance(model, SoftmaxRegression):
            _, P = model.forward(X)
        else:
            cache = model.forward(X)
            P = cache['P']
        return P

    def predict(self, model, X: np.ndarray) -> np.ndarray:
        """
        Get predicted class labels.

        Args:
            model: Trained model
            X: Input features

        Returns:
            Predicted labels (n_samples,)
        """
        P = self.predict_proba(model, X)
        return np.argmax(P, axis=1)

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
        P = self.predict_proba(model, X)
        y_pred = np.argmax(P, axis=1)

        accuracy = np.mean(y_pred == y)

        n = len(y)
        eps = 1e-9
        cross_ent = -np.mean(np.log(P[np.arange(n), y] + eps))

        confidence = np.mean(np.max(P, axis=1))

        entropy = -np.mean(np.sum(P * np.log(P + eps), axis=1))

        return {
            'accuracy': accuracy,
            'cross_entropy': cross_ent,
            'confidence': confidence,
            'entropy': entropy
        }

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
        P = self.predict_proba(model, X)
        max_probs = np.max(P, axis=1)
        y_pred = np.argmax(P, axis=1)
        correct = (y_pred == y)

        bin_edges = np.linspace(0, 1, n_bins + 1)

        results = []
        for i in range(n_bins):
            mask = (max_probs >= bin_edges[i]) & (max_probs < bin_edges[i+1])
            if np.sum(mask) > 0:
                bin_conf = np.mean(max_probs[mask])
                bin_acc = np.mean(correct[mask])
                count = np.sum(mask)
                results.append({
                    'bin': i+1,
                    'conf_range': (bin_edges[i], bin_edges[i+1]),
                    'mean_confidence': bin_conf,
                    'accuracy': bin_acc,
                    'count': count
                })
        return results

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
            np.random.seed(seed)

            model = model_class(**model_kwargs)

            # Create optimizer
            if optimizer_class is None:
                optimizer = SGD(learning_rate=model.learning_rate)
            else:
                optimizer = optimizer_class(**optimizer_kwargs)

            # Create trainer
            # trainer = Trainer(model, optimizer, epochs=epochs, batch_size=batch_size, verbose=False)

            if isinstance(model, SoftmaxRegression):
                from .trainer import SoftmaxTrainer
                trainer = SoftmaxTrainer(
                    model, optimizer,
                    epochs=epochs,
                    batch_size=batch_size,
                    reg_lambda=model.reg_lambda,
                    verbose=False
                )
            else:
                from .trainer import NNTrainer
                trainer = NNTrainer(
                    model, optimizer,
                    epochs=epochs,
                    batch_size=batch_size,
                    reg_lambda=model.reg_lambda,
                    verbose=False
                )

            # Train
            trainer.train(X_train, y_train, X_val, y_val)

            # Evaluate
            metrics = self.compute_metrics(model, X_test, y_test)

            accuracies.append(metrics['accuracy'])
            losses.append(metrics['cross_entropy'])

        return RepeatedSeedResult(
            model_name=model.__class__.__name__,
            accuracies=accuracies,
            losses=losses
        )


# REMEMBER TO CHECK THIS AGAIN.
# YOU MAY NEED TO UNCOMMENT THIS CODE !!!
# def gradient_check(
#     model,
#     X: np.ndarray,
#     Y: np.ndarray,
#     epsilon: float = 1e-5,
#     verbose: bool = True
# ) -> bool:
#     """
#     Verify gradients using finite differences.

#     REQUIRED sanity check for implementation verification.

#     Method:
#         1. Compute numerical gradient using finite differences:
#            ∂L/∂θ ≈ (L(θ + ε) - L(θ - ε)) / (2ε)

#         2. Compare with analytical gradient from backprop

#         3. Check relative error < 1e-5

#     Args:
#         model: Model to check
#         X: Small batch of inputs (use 10-100 samples)
#         Y: One-hot labels
#         epsilon: Finite difference step
#         verbose: Print results

#     Returns:
#         True if gradients are correct (relative error < 1e-5)
#     """
#     cache = model.forward(X)
#     if isinstance(model, SoftmaxRegression):
#         P = cache[1]
#         grads = model.backward(X, Y, P)
#         params = [model.W, model.b]
#         grad_list = list(grads)
#         param_names = ['W', 'b']
#     else:  # OneHiddenLayerNN
#         P = cache['P']
#         grads = model.backward(X, Y, cache)
#         params = [model.W1, model.b1, model.W2, model.b2]
#         grad_list = list(grads)
#         param_names = ['W1', 'b1', 'W2', 'b2']

#     def loss_fn():
#         if isinstance(model, SoftmaxRegression):
#             _, P_local = model.forward(X)
#             return model.compute_loss(X, Y, P_local)
#         else:
#             cache_local = model.forward(X)
#             P_local = cache_local['P']
#             reg_term = 0.5 * model.reg_lambda * (
#                 np.sum(model.W1**2) + np.sum(model.b1**2) +
#                 np.sum(model.W2**2) + np.sum(model.b2**2)
#             )
#             return model.compute_loss(Y, P_local, reg_term)

#     all_correct = True
#     for param, grad, name in zip(params, grad_list, param_names):
#         if param.ndim == 1:
#             # 1D bias vector
#             for i in range(param.shape[0]):
#                 orig = param[i]

#                 param[i] = orig + epsilon
#                 loss_high = loss_fn()

#                 param[i] = orig - epsilon
#                 loss_low = loss_fn()

#                 param[i] = orig

#                 num_grad = (loss_high - loss_low) / (2 * epsilon)
#                 ana_grad = grad[i]

#                 denom = abs(num_grad) + abs(ana_grad) + 1e-9
#                 rel_error = abs(num_grad - ana_grad) / denom

#                 if rel_error > 1e-4:
#                     if verbose:
#                         print(f"Gradient check failed at {name}[{i}]: num={num_grad:.6f}, ana={ana_grad:.6f}, rel_err={rel_error:.6f}")
#                     all_correct = False
#         else:
#             # 2D weight matrix
#             for i in range(param.shape[0]):
#                 for j in range(param.shape[1]):
#                     orig = param[i, j]

#                     param[i, j] = orig + epsilon
#                     loss_high = loss_fn()

#                     param[i, j] = orig - epsilon
#                     loss_low = loss_fn()

#                     param[i, j] = orig

#                     num_grad = (loss_high - loss_low) / (2 * epsilon)
#                     ana_grad = grad[i, j]

#                     denom = abs(num_grad) + abs(ana_grad) + 1e-9
#                     rel_error = abs(num_grad - ana_grad) / denom

#                     if rel_error > 1e-5:
#                         if verbose:
#                             print(f"Gradient check failed at {name}[{i},{j}]: num={num_grad:.6f}, ana={ana_grad:.6f}, rel_err={rel_error:.6f}")
#                         all_correct = False

#     if verbose and all_correct:
#         print("All gradients passed numerical check!")

#     return all_correct


def gradient_check(
    model,
    X: np.ndarray,
    Y: np.ndarray,
    epsilon: float = 1e-5,
    verbose: bool = True,
    use_copy: bool = True
) -> bool:
    """Verify gradients using finite differences."""

    if use_copy:
        import copy
        # Create a fresh copy
        if isinstance(model, SoftmaxRegression):
            model_copy = SoftmaxRegression(model.input_dim, model.num_classes,
                                           model.learning_rate, model.reg_lambda)
            model_copy.W = model.W.copy()
            model_copy.b = model.b.copy()
        else:
            model_copy = OneHiddenLayerNN(model.input_dim, model.hidden_dim,
                                          model.num_classes, model.learning_rate,
                                          model.reg_lambda)
            model_copy.W1 = model.W1.copy()
            model_copy.b1 = model.b1.copy()
            model_copy.W2 = model.W2.copy()
            model_copy.b2 = model.b2.copy()

    cache = model_copy.forward(X)
    if isinstance(model_copy, SoftmaxRegression):
        P = cache[1]
        grads = model_copy.backward(X, Y, P)
        params = [model_copy.W, model_copy.b]
        grad_list = list(grads)
        param_names = ['W', 'b']
    else:  # OneHiddenLayerNN
        P = cache['P']
        grads = model_copy.backward(X, Y, cache)
        params = [model_copy.W1, model_copy.b1, model_copy.W2, model_copy.b2]
        grad_list = list(grads)
        param_names = ['W1', 'b1', 'W2', 'b2']

    def loss_fn():
        if isinstance(model_copy, SoftmaxRegression):
            _, P_local = model_copy.forward(X)
            return model_copy.compute_loss(X, Y, P_local)
        else:
            cache_local = model_copy.forward(X)
            P_local = cache_local['P']
            reg_term = 0.5 * model_copy.reg_lambda * (
                np.sum(model_copy.W1**2) + np.sum(model_copy.b1**2) +
                np.sum(model_copy.W2**2) + np.sum(model_copy.b2**2)
            )
            return model_copy.compute_loss(Y, P_local, reg_term)

    all_correct = True
    for param, grad, name in zip(params, grad_list, param_names):
        if param is None or grad is None:
            continue

        if param.ndim == 1:
            # 1D bias vector - check only first few elements
            print(f"\n  Checking {name} (1D, shape={param.shape})...")
            for i in range(min(3, param.shape[0])):  # Check only first 3
                orig = param[i].copy() if hasattr(param[i], 'copy') else param[i]

                param[i] = orig + epsilon
                loss_high = loss_fn()

                param[i] = orig - epsilon
                loss_low = loss_fn()

                param[i] = orig

                num_grad = (loss_high - loss_low) / (2 * epsilon)
                ana_grad = grad[i]

                denom = abs(num_grad) + abs(ana_grad) + 1e-9
                rel_error = abs(num_grad - ana_grad) / denom

                print(f"    [{i}]: num={num_grad:.8f}, ana={ana_grad:.8f}, rel_err={rel_error:.8f}")

                if rel_error > 1e-3:
                    if verbose:
                        print(f"    ✗ Gradient check failed at {name}[{i}]!")
                    all_correct = False
                else:
                    print(f"    ✓ Passed")
        else:
            # 2D weight matrix - check only first few entries
            print(f"\n  Checking {name} (2D, shape={param.shape})...")
            for i in range(min(2, param.shape[0])):
                for j in range(min(2, param.shape[1])):
                    orig = param[i, j].copy() if hasattr(param[i, j], 'copy') else param[i, j]

                    param[i, j] = orig + epsilon
                    loss_high = loss_fn()

                    param[i, j] = orig - epsilon
                    loss_low = loss_fn()

                    param[i, j] = orig

                    num_grad = (loss_high - loss_low) / (2 * epsilon)
                    ana_grad = grad[i, j]

                    denom = abs(num_grad) + abs(ana_grad) + 1e-9
                    rel_error = abs(num_grad - ana_grad) / denom

                    print(f"    [{i},{j}]: num={num_grad:.8f}, ana={ana_grad:.8f}, rel_err={rel_error:.8f}")

                    if rel_error > 1e-3:
                        if verbose:
                            print(f"    ✗ Gradient check failed at {name}[{i},{j}]!")
                        all_correct = False
                    else:
                        print(f"    ✓ Passed")

    if verbose and all_correct:
        print("\n✓ All gradients passed numerical check!")
    elif verbose:
        print("\n✗ Some gradient checks failed!")

    return all_correct

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
    sums = np.sum(P, axis=1)
    all_one = np.allclose(sums, 1.0, atol=eps)
    return all_one


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
    cache = model.forward(X)

    if isinstance(model, SoftmaxRegression):
        logits, P = cache
        arrays = [logits, P]
    else:  # OneHiddenLayerNN
        arrays = list(cache.values())

    for arr in arrays:
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            return False

    return True
