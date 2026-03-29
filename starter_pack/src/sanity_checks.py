"""
Sanity checks for model implementation verification.
"""

import numpy as np
from typing import Tuple, Callable

from .models import SoftmaxRegression, OneHiddenLayerNN
from .trainer import SoftmaxTrainer, NNTrainer
from .optimizers import SGD
from .evaluation import gradient_check, check_probability_sum, check_nan_inf


def test_loss_decreases_on_tiny_subset(
    model,
    trainer_class,
    X_tiny: np.ndarray,
    y_tiny: np.ndarray,
    steps: int = 10
) -> bool:
    """
    Verify that loss decreases on a tiny subset after a few updates.

    Args:
        model: Model to test
        trainer_class: SoftmaxTrainer or NNTrainer
        X_tiny: Tiny batch of inputs (e.g., 5-10 samples)
        y_tiny: Corresponding labels
        steps: Number of training steps

    Returns:
        True if loss decreased
    """
    # Create trainer with tiny batch size = full dataset
    optimizer = SGD(learning_rate=model.learning_rate)
    trainer = trainer_class(
        model, optimizer,
        epochs=1,  # We'll control steps manually
        batch_size=len(X_tiny),
        reg_lambda=model.reg_lambda,
        verbose=False
    )

    # Get initial loss
    initial_loss, _ = trainer.evaluate(X_tiny, y_tiny)

    # Train for specified steps
    for _ in range(steps):
        trainer.train_epoch(X_tiny, y_tiny)

    # Get final loss
    final_loss, _ = trainer.evaluate(X_tiny, y_tiny)

    print(f"  Loss decreased: {initial_loss:.6f} → {final_loss:.6f}")
    return final_loss < initial_loss


def test_overfitting_small_subset(
    model_class,
    model_kwargs: dict,
    X_subset: np.ndarray,
    y_subset: np.ndarray,
    trainer_class,
    epochs: int = 200
) -> Tuple[bool, float]:
    """
    Verify model can overfit a very small subset (reach 100% accuracy).

    Args:
        model_class: SoftmaxRegression or OneHiddenLayerNN
        model_kwargs: Arguments for model constructor
        X_subset: Tiny subset (e.g., 10-20 samples)
        y_subset: Corresponding labels
        trainer_class: SoftmaxTrainer or NNTrainer
        epochs: Number of epochs to train

    Returns:
        Tuple of (overfit_success, final_accuracy)
    """
    np.random.seed(42)
    model = model_class(**model_kwargs)
    optimizer = SGD(learning_rate=model.learning_rate)
    trainer = trainer_class(
        model, optimizer,
        epochs=epochs,
        batch_size=len(X_subset),  # Full batch
        reg_lambda=model.reg_lambda,
        verbose=False
    )

    # Train
    history = trainer.train(X_subset, y_subset, X_subset, y_subset)

    final_acc = history.train_accuracies[-1]
    overfit_success = final_acc >= 0.99

    print(f"  Overfitting: Final accuracy = {final_acc:.4f} (success: {overfit_success})")
    return overfit_success, final_acc


def run_all_sanity_checks(moons_data, digits_data):
    """
    Run all 5 required sanity checks.

    Should be called BEFORE main experiments in main.py.
    """
    print("\n" + "="*60)
    print("RUNNING SANITY CHECKS")
    print("="*60)

    X_moons = moons_data['X_train']
    y_moons = moons_data['y_train']

    # ============================================
    # 1. Gradient check on tiny batch
    # ============================================
    print("\n[1/5] Gradient Check...")
    X_tiny = X_moons[:10]
    y_tiny = y_moons[:10]

    # One-hot encode
    from .data_loader import DataLoader
    loader = DataLoader()
    Y_tiny = loader.one_hot_encode(y_tiny, num_classes=2)

    model = OneHiddenLayerNN(input_dim=2, hidden_dim=4, num_classes=2)
    success = gradient_check(model, X_tiny, Y_tiny, verbose=False)
    print(f"  {'✓' if success else '✗'} Gradient check {'passed' if success else 'failed'}")

    # ============================================
    # 2. Loss decreases on tiny subset
    # ============================================
    print("\n[2/5] Loss Decrease Check...")
    model = OneHiddenLayerNN(input_dim=2, hidden_dim=4, num_classes=2)
    success = test_loss_decreases_on_tiny_subset(
        model, NNTrainer, X_tiny, y_tiny, steps=10
    )
    print(f"  {'✓' if success else '✗'} Loss decreased after updates")

    # ============================================
    # 3. Overfitting very small subset
    # ============================================
    print("\n[3/5] Overfitting Check...")
    # Use only 10 samples
    X_very_small = X_moons[:10]
    y_very_small = y_moons[:10]

    success, acc = test_overfitting_small_subset(
        OneHiddenLayerNN,
        {'input_dim': 2, 'hidden_dim': 32, 'num_classes': 2},
        X_very_small, y_very_small,
        NNTrainer,
        epochs=200
    )
    print(f"  {'✓' if success else '✗'} Model overfit small subset (acc={acc:.4f})")

    # ============================================
    # 4. Probabilities sum to 1
    # ============================================
    print("\n[4/5] Probability Sum Check...")
    model = OneHiddenLayerNN(input_dim=2, hidden_dim=4, num_classes=2)
    cache = model.forward(X_tiny)
    success = check_probability_sum(cache['P'])
    print(f"  {'✓' if success else '✗'} Probabilities sum to 1")

    # ============================================
    # 5. No NaN/Inf
    # ============================================
    print("\n[5/5] NaN/Inf Check...")
    success = check_nan_inf(model, X_tiny)
    print(f"  {'✓' if success else '✗'} No NaN or Inf values")

    print("\n" + "="*60)
    print("SANITY CHECKS COMPLETE")
    print("="*60)

    return True