"""
Main experiment script for Math4AI Capstone Project.

This script runs the required experiments:
1. Softmax Regression (baseline) vs One-Hidden-Layer Neural Network
2. Three datasets: Linear Gaussian, Moons, Digits
3. Required ablations: capacity, optimizer study, failure case
4. Track A or Track B advanced analysis

Usage:
    python main.py --experiment all
    python main.py --experiment linear_gaussian
    python main.py --experiment moons
    python main.py --experiment digits
    python main.py --experiment ablations
    python main.py --experiment track_a
    python main.py --experiment track_b
"""

import numpy as np
import argparse
import os
import json
from datetime import datetime

from starter_pack.src.data_loader import DataLoader
from starter_pack.src.models import SoftmaxRegression, OneHiddenLayerNN
from starter_pack.src.optimizers import SGD, Momentum, Adam, create_optimizer
from starter_pack.src.trainer import SoftmaxTrainer, NNTrainer
from starter_pack.src.evaluation import Evaluator, RepeatedSeedResult
from starter_pack.src.visualization import (
    plot_decision_boundary,
    plot_training_dynamics,
    plot_confidence_vs_accuracy,
    plot_optimizer_comparison,
    plot_capacity_ablation_boundaries,
    plot_repeated_seed_results,
    plot_pca_scree,
    plot_pca_2d,
    save_results_table
)


# Default hyperparameters (from PDF protocol)
DEFAULTS = {
    'hidden_width': 32,
    'reg_lambda': 1e-4,
    'batch_size': 64,
    'epochs': 200,
    'lr_softmax': 0.05,
    'lr_sgd': 0.05,
    'lr_momentum': 0.05,
    'momentum': 0.9,
    'lr_adam': 0.001,
    'seeds': [42, 123, 456, 789, 1000]
}


def setup_directories():
    """Create necessary output directories."""
    dirs = ['starter_pack/figures', 'starter_pack/results']
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def check_implementation(result, name, filename, function_name):
    """
    Check if a function has been implemented.
    
    Args:
        result: The return value from the function
        name: Human-readable name of what's missing
        filename: Source file where function is located
        function_name: Name of the unimplemented function
    """
    if result is None:
        print(f"""
{'!'*60}
[!] IMPLEMENTATION MISSING: {name}
{'!'*60}

Location: starter_pack/src/{filename}
Function: {function_name}()

This function needs to be implemented before running experiments.

How to fix:
1. Open: starter_pack/src/{filename}
2. Find: def {function_name}()
3. Remove the 'pass' statements and implement the code
4. Look for TODO comments for guidance

{'!'*60}
""")
        return False
    return True


def check_model_implementation():
    """
    Check if all core model implementations are complete.
    
    Returns:
        List of missing implementations
    """
    missing = []
    
    print("\n" + "="*60)
    print("CHECKING IMPLEMENTATIONS...")
    print("="*60)
    
    # Test DataLoader
    print("\n[1/6] Checking DataLoader...")
    loader = DataLoader()
    test_data = loader.load_synthetic('linear_gaussian')
    if test_data is None or test_data.get('X_train') is None:
        missing.append(("DataLoader.load_synthetic()", "data_loader.py", "load_synthetic"))
        print("    [X] DataLoader.load_synthetic() - NOT IMPLEMENTED")
    else:
        print("    [OK] DataLoader.load_synthetic()")
    
    # Test Softmax Regression forward pass
    print("\n[2/6] Checking SoftmaxRegression...")
    try:
        model = SoftmaxRegression(input_dim=2, num_classes=2)
        X_test = np.random.randn(5, 2)
        result = model.forward(X_test)
        if result is None or result[0] is None:
            missing.append(("SoftmaxRegression.forward()", "models.py", "forward"))
            print("    [X] SoftmaxRegression.forward() - NOT IMPLEMENTED")
        else:
            print("    [OK] SoftmaxRegression.forward()")
    except:
        missing.append(("SoftmaxRegression", "models.py", "SoftmaxRegression class"))
        print("    [X] SoftmaxRegression - NOT IMPLEMENTED")
    
    # Test Softmax Regression backward
    print("\n[3/6] Checking SoftmaxRegression backward...")
    try:
        model = SoftmaxRegression(input_dim=2, num_classes=2)
        X_test = np.random.randn(5, 2)
        logits, P = model.forward(X_test)
        Y_test = np.zeros((5, 2))
        Y_test[:, 0] = 1
        result = model.backward(X_test, Y_test, P)
        if result is None or result[0] is None:
            missing.append(("SoftmaxRegression.backward()", "models.py", "backward"))
            print("    [X] SoftmaxRegression.backward() - NOT IMPLEMENTED")
        else:
            print("    [OK] SoftmaxRegression.backward()")
    except:
        missing.append(("SoftmaxRegression.backward()", "models.py", "backward"))
        print("    [X] SoftmaxRegression.backward() - NOT IMPLEMENTED")
    
    # Test Neural Network
    print("\n[4/6] Checking OneHiddenLayerNN...")
    try:
        model = OneHiddenLayerNN(input_dim=2, hidden_dim=4, num_classes=2)
        X_test = np.random.randn(5, 2)
        result = model.forward(X_test)
        if result is None or result.get('P') is None:
            missing.append(("OneHiddenLayerNN.forward()", "models.py", "OneHiddenLayerNN.forward"))
            print("    [X] OneHiddenLayerNN.forward() - NOT IMPLEMENTED")
        else:
            print("    [OK] OneHiddenLayerNN.forward()")
    except:
        missing.append(("OneHiddenLayerNN", "models.py", "OneHiddenLayerNN class"))
        print("    [X] OneHiddenLayerNN - NOT IMPLEMENTED")
    
    # Test Neural Network backward (BACKPROP)
    print("\n[5/6] Checking OneHiddenLayerNN backward (BACKPROP)...")
    try:
        model = OneHiddenLayerNN(input_dim=2, hidden_dim=4, num_classes=2)
        X_test = np.random.randn(5, 2)
        cache = model.forward(X_test)
        Y_test = np.zeros((5, 2))
        Y_test[:, 0] = 1
        result = model.backward(X_test, Y_test, cache)
        if result is None or result[0] is None:
            missing.append(("OneHiddenLayerNN.backward() [BACKPROP]", "models.py", "OneHiddenLayerNN.backward"))
            print("    [X] OneHiddenLayerNN.backward() [BACKPROP] - NOT IMPLEMENTED")
        else:
            print("    [OK] OneHiddenLayerNN.backward() [BACKPROP]")
    except:
        missing.append(("OneHiddenLayerNN.backward() [BACKPROP]", "models.py", "OneHiddenLayerNN.backward"))
        print("    [X] OneHiddenLayerNN.backward() [BACKPROP] - NOT IMPLEMENTED")
    
    # Test Trainer
    print("\n[6/6] Checking Trainer...")
    try:
        model = SoftmaxRegression(input_dim=2, num_classes=2)
        trainer = SoftmaxTrainer(model, SGD(0.05), epochs=1, verbose=False)
        X_train = np.random.randn(10, 2)
        y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        result = trainer.train_epoch(X_train, y_train)
        if result is None or result[0] is None:
            missing.append(("SoftmaxTrainer.train_epoch()", "trainer.py", "SoftmaxTrainer.train_epoch"))
            print("    [X] SoftmaxTrainer.train_epoch() - NOT IMPLEMENTED")
        else:
            print("    [OK] SoftmaxTrainer.train_epoch()")
    except:
        missing.append(("SoftmaxTrainer", "trainer.py", "SoftmaxTrainer class"))
        print("    [X] SoftmaxTrainer - NOT IMPLEMENTED")
    
    # Summary
    print("\n" + "="*60)
    if missing:
        print(f"[!] FOUND {len(missing)} MISSING IMPLEMENTATION(S)")
        print("="*60)
        for i, (name, file, func) in enumerate(missing, 1):
            print(f"\n{i}. {name}")
            print(f"   File: starter_pack/src/{file}")
            print(f"   Function: {func}()")
        print("\n" + "="*60)
        print("Please implement the missing functions before running experiments.")
        print("="*60)
        return missing
    else:
        print("[OK] ALL IMPLEMENTATIONS COMPLETE!")
        print("="*60)
        return []


def load_data():
    """Load all datasets."""
    print("\n" + "="*60)
    print("LOADING DATA...")
    print("="*60)
    
    loader = DataLoader()
    
    print("\n[1/3] Loading Linear Gaussian data...")
    linear_data = loader.load_synthetic('linear_gaussian')
    if not check_implementation(linear_data, "DataLoader.load_synthetic()", "data_loader.py", "load_synthetic"):
        linear_data = {'X_train': None, 'X_val': None, 'X_test': None, 'y_train': None, 'y_val': None, 'y_test': None}
    
    print("\n[2/3] Loading Moons data...")
    moons_data = loader.load_synthetic('moons')
    if not check_implementation(moons_data, "DataLoader.load_synthetic()", "data_loader.py", "load_synthetic"):
        moons_data = {'X_train': None, 'X_val': None, 'X_test': None, 'y_train': None, 'y_val': None, 'y_test': None}
    
    print("\n[3/3] Loading Digits data...")
    digits_data = loader.load_digits()
    if not check_implementation(digits_data, "DataLoader.load_digits()", "data_loader.py", "load_digits"):
        digits_data = (None, None, None, None, None, None)
    
    print("\n[OK] Data loading complete!")
    return linear_data, moons_data, digits_data


def run_synthetic_experiment(
    name: str,
    data: dict,
    hidden_width: int = 8,
    epochs: int = 200
):
    """
    Run experiment on synthetic dataset.
    
    Required: decision boundary plots.
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*60}")
    
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    
    n_classes = len(np.unique(y_train))
    input_dim = X_train.shape[1]
    
    results = {}
    
    # Softmax Regression
    print("\nTraining Softmax Regression...")
    np.random.seed(42)
    softmax_model = SoftmaxRegression(
        input_dim=input_dim,
        num_classes=n_classes,
        learning_rate=DEFAULTS['lr_softmax'],
        reg_lambda=DEFAULTS['reg_lambda']
    )
    softmax_optimizer = SGD(learning_rate=DEFAULTS['lr_softmax'])
    softmax_trainer = SoftmaxTrainer(
        softmax_model, softmax_optimizer,
        epochs=epochs, batch_size=DEFAULTS['batch_size'],
        reg_lambda=DEFAULTS['reg_lambda'],
        verbose=False
    )
    softmax_history = softmax_trainer.train(X_train, y_train, X_val, y_val)
    
    evaluator = Evaluator()
    softmax_metrics = evaluator.compute_metrics(softmax_model, X_test, y_test)
    results['softmax'] = {
        'accuracy': softmax_metrics['accuracy'],
        'loss': softmax_metrics['cross_entropy'],
        'history': softmax_history
    }
    print(f"Softmax - Test Accuracy: {softmax_metrics['accuracy']:.4f}")
    
    # Neural Network
    print("\nTraining Neural Network...")
    np.random.seed(42)
    nn_model = OneHiddenLayerNN(
        input_dim=input_dim,
        hidden_dim=hidden_width,
        num_classes=n_classes,
        learning_rate=DEFAULTS['lr_sgd'],
        reg_lambda=DEFAULTS['reg_lambda']
    )
    nn_optimizer = SGD(learning_rate=DEFAULTS['lr_sgd'])
    nn_trainer = NNTrainer(
        nn_model, nn_optimizer,
        epochs=epochs, batch_size=DEFAULTS['batch_size'],
        reg_lambda=DEFAULTS['reg_lambda'],
        verbose=False
    )
    nn_history = nn_trainer.train(X_train, y_train, X_val, y_val)
    
    nn_metrics = evaluator.compute_metrics(nn_model, X_test, y_test)
    results['nn'] = {
        'accuracy': nn_metrics['accuracy'],
        'loss': nn_metrics['cross_entropy'],
        'history': nn_history
    }
    print(f"NN - Test Accuracy: {nn_metrics['accuracy']:.4f}")
    
    # Decision boundary plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    plot_decision_boundary(
        softmax_model, X_test, y_test,
        title=f'Softmax Regression\nAccuracy: {softmax_metrics["accuracy"]:.4f}',
        ax=axes[0]
    )
    
    plot_decision_boundary(
        nn_model, X_test, y_test,
        title=f'Neural Network (h={hidden_width})\nAccuracy: {nn_metrics["accuracy"]:.4f}',
        ax=axes[1]
    )
    
    plt.suptitle(f'Decision Boundaries: {name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'starter_pack/figures/decision_boundary_{name}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Training dynamics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    plot_training_dynamics(softmax_history, title='Softmax Training', ax=axes[0])
    plot_training_dynamics(nn_history, title='NN Training', ax=axes[1])
    
    plt.savefig(f'starter_pack/figures/training_dynamics_{name}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return results


def run_digits_experiment(digits_data, track: str = 'base'):
    """
    Run experiment on digits benchmark.
    
    Required:
    - Same preprocessing and split
    - Report accuracy and cross-entropy
    - Use validation cross-entropy for model selection
    - Best validation checkpoint
    - Repeated-seed evaluation (5 seeds)
    """
    print(f"\n{'='*60}")
    print("EXPERIMENT: Digits Benchmark")
    print(f"{'='*60}")
    
    X, y = digits_data['X'], digits_data['y']
    train_idx, val_idx, test_idx = digits_data['train_idx'], digits_data['val_idx'], digits_data['test_idx']
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y))
    
    results = {}
    all_histories = {}
    
    # Softmax Regression
    print("\nTraining Softmax Regression on Digits...")
    np.random.seed(42)
    softmax_model = SoftmaxRegression(
        input_dim=input_dim,
        num_classes=num_classes,
        learning_rate=DEFAULTS['lr_softmax'],
        reg_lambda=DEFAULTS['reg_lambda']
    )
    softmax_optimizer = SGD(learning_rate=DEFAULTS['lr_softmax'])
    softmax_trainer = SoftmaxTrainer(
        softmax_model, softmax_optimizer,
        epochs=DEFAULTS['epochs'], batch_size=DEFAULTS['batch_size'],
        reg_lambda=DEFAULTS['reg_lambda'],
        verbose=False
    )
    softmax_history = softmax_trainer.train(X_train, y_train, X_val, y_val)
    
    evaluator = Evaluator()
    softmax_metrics = evaluator.compute_metrics(softmax_model, X_test, y_test)
    results['softmax'] = {
        'test_accuracy': softmax_metrics['accuracy'],
        'test_loss': softmax_metrics['cross_entropy'],
        'history': softmax_history
    }
    all_histories['softmax'] = softmax_history
    print(f"Softmax - Test Accuracy: {softmax_metrics['accuracy']:.4f}, Loss: {softmax_metrics['cross_entropy']:.4f}")
    
    # Neural Network
    print("\nTraining Neural Network on Digits...")
    np.random.seed(42)
    nn_model = OneHiddenLayerNN(
        input_dim=input_dim,
        hidden_dim=DEFAULTS['hidden_width'],
        num_classes=num_classes,
        learning_rate=DEFAULTS['lr_sgd'],
        reg_lambda=DEFAULTS['reg_lambda']
    )
    nn_optimizer = SGD(learning_rate=DEFAULTS['lr_sgd'])
    nn_trainer = NNTrainer(
        nn_model, nn_optimizer,
        epochs=DEFAULTS['epochs'], batch_size=DEFAULTS['batch_size'],
        reg_lambda=DEFAULTS['reg_lambda'],
        verbose=False
    )
    nn_history = nn_trainer.train(X_train, y_train, X_val, y_val)
    
    nn_metrics = evaluator.compute_metrics(nn_model, X_test, y_test)
    results['nn'] = {
        'test_accuracy': nn_metrics['accuracy'],
        'test_loss': nn_metrics['cross_entropy'],
        'history': nn_history
    }
    all_histories['nn'] = nn_history
    print(f"NN - Test Accuracy: {nn_metrics['accuracy']:.4f}, Loss: {nn_metrics['cross_entropy']:.4f}")
    
    # Training dynamics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_training_dynamics(softmax_history, title='Softmax on Digits', ax=axes[0])
    plot_training_dynamics(nn_history, title='NN on Digits', ax=axes[1])
    plt.savefig('starter_pack/figures/training_dynamics_digits.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Repeated-seed evaluation
    print("\nRunning repeated-seed evaluation (5 seeds)...")
    
    softmax_seeds = []
    nn_seeds = []
    
    for seed in DEFAULTS['seeds']:
        np.random.seed(seed)
        
        # Softmax
        sm = SoftmaxRegression(input_dim, num_classes, DEFAULTS['lr_softmax'], DEFAULTS['reg_lambda'])
        sm_opt = SGD(DEFAULTS['lr_softmax'])
        sm_trainer = SoftmaxTrainer(sm, sm_opt, DEFAULTS['epochs'], DEFAULTS['batch_size'], 
                                    DEFAULTS['reg_lambda'], verbose=False)
        sm_trainer.train(X_train, y_train, X_val, y_val)
        sm_metrics = evaluator.compute_metrics(sm, X_test, y_test)
        softmax_seeds.append({
            'accuracy': sm_metrics['accuracy'],
            'loss': sm_metrics['cross_entropy']
        })
        
        # NN
        nn = OneHiddenLayerNN(input_dim, DEFAULTS['hidden_width'], num_classes, 
                              DEFAULTS['lr_sgd'], DEFAULTS['reg_lambda'])
        nn_opt = SGD(DEFAULTS['lr_sgd'])
        nn_trainer = NNTrainer(nn, nn_opt, DEFAULTS['epochs'], DEFAULTS['batch_size'],
                               DEFAULTS['reg_lambda'], verbose=False)
        nn_trainer.train(X_train, y_train, X_val, y_val)
        nn_metrics = evaluator.compute_metrics(nn, X_test, y_test)
        nn_seeds.append({
            'accuracy': nn_metrics['accuracy'],
            'loss': nn_metrics['cross_entropy']
        })
    
    results['softmax_seeds'] = softmax_seeds
    results['nn_seeds'] = nn_seeds
    
    # Print summary
    print("\n" + "="*60)
    print("REPEATED SEED RESULTS")
    print("="*60)
    
    sm_acc = [s['accuracy'] for s in softmax_seeds]
    sm_loss = [s['loss'] for s in softmax_seeds]
    nn_acc = [s['accuracy'] for s in nn_seeds]
    nn_loss = [s['loss'] for s in nn_seeds]
    
    print(f"Softmax: Acc={np.mean(sm_acc):.4f}±{np.std(sm_acc, ddof=1):.4f}, "
          f"Loss={np.mean(sm_loss):.4f}±{np.std(sm_loss, ddof=1):.4f}")
    print(f"NN:      Acc={np.mean(nn_acc):.4f}±{np.std(nn_acc, ddof=1):.4f}, "
          f"Loss={np.mean(nn_loss):.4f}±{np.std(nn_loss, ddof=1):.4f}")
    
    return results, all_histories


def run_optimizer_study(digits_data):
    """
    Required ablation: Compare SGD, Momentum, Adam on NN.
    """
    print(f"\n{'='*60}")
    print("EXPERIMENT: Optimizer Study")
    print(f"{'='*60}")
    
    X, y = digits_data['X'], digits_data['y']
    X_train, y_train = X[digits_data['train_idx']], y[digits_data['train_idx']]
    X_val, y_val = X[digits_data['val_idx']], y[digits_data['val_idx']]
    
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y))
    
    optimizers_config = [
        ('SGD', SGD(learning_rate=DEFAULTS['lr_sgd'])),
        ('Momentum', Momentum(learning_rate=DEFAULTS['lr_momentum'], momentum=DEFAULTS['momentum'])),
        ('Adam', Adam(learning_rate=DEFAULTS['lr_adam']))
    ]
    
    histories = {}
    
    for opt_name, optimizer in optimizers_config:
        print(f"\nTraining with {opt_name}...")
        np.random.seed(42)
        
        model = OneHiddenLayerNN(
            input_dim=input_dim,
            hidden_dim=DEFAULTS['hidden_width'],
            num_classes=num_classes,
            learning_rate=optimizer.learning_rate,
            reg_lambda=DEFAULTS['reg_lambda']
        )
        
        trainer = NNTrainer(
            model, optimizer,
            epochs=DEFAULTS['epochs'],
            batch_size=DEFAULTS['batch_size'],
            reg_lambda=DEFAULTS['reg_lambda'],
            verbose=False
        )
        
        history = trainer.train(X_train, y_train, X_val, y_val)
        histories[opt_name] = history
        
        print(f"{opt_name} - Final Val Acc: {history.val_accuracies[-1]:.4f}")
    
    plot_optimizer_comparison(histories, title='Optimizer Comparison on Digits')
    plt.savefig('starter_pack/figures/optimizer_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return histories


def run_capacity_ablation(moons_data):
    """
    Required ablation: Compare hidden widths {2, 8, 32} on moons.
    """
    print(f"\n{'='*60}")
    print("EXPERIMENT: Capacity Ablation")
    print(f"{'='*60}")
    
    X_train, y_train = moons_data['X_train'], moons_data['y_train']
    X_val, y_val = moons_data['X_val'], moons_data['y_val']
    X_test, y_test = moons_data['X_test'], moons_data['y_test']
    
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    hidden_widths = [2, 8, 32]
    histories = {}
    models = {}
    
    for width in hidden_widths:
        print(f"\nTraining with hidden_width={width}...")
        np.random.seed(42)
        
        model = OneHiddenLayerNN(
            input_dim=input_dim,
            hidden_dim=width,
            num_classes=num_classes,
            learning_rate=DEFAULTS['lr_sgd'],
            reg_lambda=DEFAULTS['reg_lambda']
        )
        
        optimizer = SGD(learning_rate=DEFAULTS['lr_sgd'])
        trainer = NNTrainer(
            model, optimizer,
            epochs=200,
            batch_size=DEFAULTS['batch_size'],
            reg_lambda=DEFAULTS['reg_lambda'],
            verbose=False
        )
        
        history = trainer.train(X_train, y_train, X_val, y_val)
        histories[width] = history
        models[width] = model
        
        test_acc = np.mean(np.argmax(trainer.model.predict(X_test)[1], axis=1) == y_test)
        print(f"Width={width} - Test Accuracy: {test_acc:.4f}")
    
    # Decision boundaries
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, width in enumerate(hidden_widths):
        plot_decision_boundary(
            models[width], X_test, y_test,
            title=f'hidden_width={width}',
            ax=axes[i]
        )
    
    plt.suptitle('Capacity Ablation: Decision Boundaries', fontsize=16)
    plt.tight_layout()
    plt.savefig('starter_pack/figures/capacity_ablation_boundaries.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Loss curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, width in enumerate(hidden_widths):
        axes[i].plot(histories[width].val_losses, 'b-', linewidth=2)
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Validation Loss')
        axes[i].set_title(f'hidden_width={width}')
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Capacity Ablation: Training Curves', fontsize=16)
    plt.tight_layout()
    plt.savefig('starter_pack/figures/capacity_ablation_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return histories, models


def run_track_a(digits_data):
    """
    Track A: PCA/SVD and input geometry analysis.
    
    Required:
    - Scree plot
    - 2D PCA visualization
    - Comparison at PCA dimensions {10, 20, 40}
    """
    print(f"\n{'='*60}")
    print("TRACK A: PCA/SVD Analysis")
    print(f"{'='*60}")
    
    X, y = digits_data['X'], digits_data['y']
    X_train, y_train = X[digits_data['train_idx']], y[digits_data['train_idx']]
    X_val, y_val = X[digits_data['val_idx']], y[digits_data['val_idx']]
    X_test, y_test = X[digits_data['test_idx']], y[digits_data['test_idx']]
    
    # PCA
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    eigenvalues = S ** 2 / (len(S) - 1)
    
    plot_pca_scree(eigenvalues, title='PCA Scree Plot (Digits)')
    plt.savefig('starter_pack/figures/track_a_scree.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 2D PCA visualization
    X_pca_2d = U[:, :2] * S[:2]
    plot_pca_2d(X_pca_2d, y, title='PCA 2D Visualization (Digits)')
    plt.savefig('starter_pack/figures/track_a_pca2d.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Classification at different PCA dimensions
    dimensions = [10, 20, 40]
    results = {}
    
    print("\nClassification at different PCA dimensions:")
    
    for dim in dimensions:
        X_proj = U[:, :dim] * S[:dim]
        
        X_tr = X_proj[digits_data['train_idx']]
        X_vl = X_proj[digits_data['val_idx']]
        X_ts = X_proj[digits_data['test_idx']]
        
        np.random.seed(42)
        model = SoftmaxRegression(dim, 10, DEFAULTS['lr_softmax'], DEFAULTS['reg_lambda'])
        opt = SGD(DEFAULTS['lr_softmax'])
        trainer = SoftmaxTrainer(model, opt, 200, 64, DEFAULTS['reg_lambda'], verbose=False)
        trainer.train(X_tr, y_train, X_vl, y_val)
        
        evaluator = Evaluator()
        metrics = evaluator.compute_metrics(model, X_ts, y_test)
        
        results[dim] = metrics
        print(f"  dim={dim}: Acc={metrics['accuracy']:.4f}, Loss={metrics['cross_entropy']:.4f}")
    
    return results


def run_track_b(digits_data):
    """
    Track B: Prediction confidence and reliability.
    
    Required:
    - Confidence vs accuracy bins (5 bins)
    - Compare correct vs incorrect predictions
    """
    print(f"\n{'='*60}")
    print("TRACK B: Confidence and Reliability")
    print(f"{'='*60}")
    
    X, y = digits_data['X'], digits_data['y']
    X_test, y_test = X[digits_data['test_idx']], y[digits_data['test_idx']]
    
    # Train both models
    X_train, y_train = X[digits_data['train_idx']], y[digits_data['train_idx']]
    X_val, y_val = X[digits_data['val_idx']], y[digits_data['val_idx']]
    
    # Softmax
    np.random.seed(42)
    softmax = SoftmaxRegression(64, 10, DEFAULTS['lr_softmax'], DEFAULTS['reg_lambda'])
    SoftmaxTrainer(softmax, SGD(DEFAULTS['lr_softmax']), 200, 64, 
                   DEFAULTS['reg_lambda'], verbose=False).train(X_train, y_train, X_val, y_val)
    
    # NN
    np.random.seed(42)
    nn = OneHiddenLayerNN(64, DEFAULTS['hidden_width'], 10, DEFAULTS['lr_sgd'], DEFAULTS['reg_lambda'])
    NNTrainer(nn, SGD(DEFAULTS['lr_sgd']), 200, 64,
              DEFAULTS['reg_lambda'], verbose=False).train(X_train, y_train, X_val, y_val)
    
    evaluator = Evaluator()
    
    print("\nConfidence vs Accuracy Analysis:")
    
    for name, model in [('Softmax', softmax), ('NN', nn)]:
        P = evaluator.predict_proba(model, X_test)
        conf_bins = evaluator.confidence_by_bin(model, X_test, y_test, n_bins=5)
        
        print(f"\n{name}:")
        print(f"{'Bin':<6} {'Conf Range':<15} {'Mean Conf':<12} {'Accuracy':<10} {'Count':<8}")
        print("-" * 55)
        for b in conf_bins:
            print(f"{b['bin']:<6} {str(b['conf_range']):<15} {b['mean_confidence']:<12.4f} "
                  f"{b['accuracy']:<10.4f} {b['count']:<8}")
        
        plot_confidence_vs_accuracy(conf_bins, title=f'{name}: Confidence vs Accuracy')
        plt.savefig(f'starter_pack/figures/track_b_confidence_{name.lower()}.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # Correct vs Incorrect analysis
    print("\nCorrect vs Incorrect Prediction Analysis:")
    
    for name, model in [('Softmax', softmax), ('NN', nn)]:
        P = evaluator.predict_proba(model, X_test)
        y_pred = np.argmax(P, axis=1)
        correct_mask = (y_pred == y_test)
        
        max_probs = np.max(P, axis=1)
        entropy = -np.sum(P * np.log(P + 1e-9), axis=1)
        
        print(f"\n{name}:")
        print(f"  Correct predictions:   mean_conf={max_probs[correct_mask].mean():.4f}, "
              f"mean_entropy={entropy[correct_mask].mean():.4f}")
        print(f"  Incorrect predictions: mean_conf={max_probs[~correct_mask].mean():.4f}, "
              f"mean_entropy={entropy[~correct_mask].mean():.4f}")


def run_failure_case_analysis(moons_data):
    """
    Required: Analyze one failure case.
    
    This could be under-capacity, optimizer issues, instability, or overfitting.
    """
    print(f"\n{'='*60}")
    print("EXPERIMENT: Failure Case Analysis")
    print(f"{'='*60}")
    
    X_train, y_train = moons_data['X_train'], moons_data['y_train']
    X_val, y_val = moons_data['X_val'], moons_data['y_val']
    X_test, y_test = moons_data['X_test'], moons_data['y_test']
    
    # Failure case: Very small hidden width (capacity insufficient)
    print("\nFailure Case: hidden_width=1 (severe under-capacity)")
    
    np.random.seed(42)
    model = OneHiddenLayerNN(2, 1, 2, 0.05, 1e-4)
    
    from starter_pack.src.optimizers import SGD
    optimizer = SGD(0.05)
    trainer = NNTrainer(model, optimizer, epochs=200, batch_size=64, reg_lambda=1e-4, verbose=False)
    history = trainer.train(X_train, y_train, X_val, y_val)
    
    evaluator = Evaluator()
    metrics = evaluator.compute_metrics(model, X_test, y_test)
    
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Loss: {metrics['cross_entropy']:.4f}")
    
    # Analysis: why did it fail?
    print("\nAnalysis:")
    print("- With hidden_width=1, the network cannot represent nonlinear boundaries")
    print("- The single hidden unit acts as a linear transformation")
    print("- This demonstrates when additional complexity is necessary")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    plot_decision_boundary(
        model, X_test, y_test,
        title=f'Failed Model (h=1)\nAcc={metrics["accuracy"]:.4f}',
        ax=axes[0]
    )
    
    axes[1].plot(history.val_losses, 'r-', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Loss')
    axes[1].set_title('Training Curve (Failure)')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Failure Case: Under-Capacity Network', fontsize=16)
    plt.tight_layout()
    plt.savefig('starter_pack/figures/failure_case_undercapacity.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return history, metrics


def main():
    parser = argparse.ArgumentParser(description='Math4AI Capstone Experiments')
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['all', 'linear_gaussian', 'moons', 'digits', 
                               'ablations', 'track_a', 'track_b', 'failure', 'check'],
                       help='Which experiment to run')
    parser.add_argument('--track', type=str, default='a',
                       choices=['a', 'b'],
                       help='Advanced track (a or b)')
    args = parser.parse_args()
    
    print("="*60)
    print("Math4AI Capstone: From Linear Scores to Single Hidden Layer")
    print("="*60)
    
    # Check implementation status
    missing = check_model_implementation()
    
    # If 'check' argument, just show status and exit
    if args.experiment == 'check':
        print("\n[!] Implementation check complete.")
        return
    
    # If there are missing implementations, ask user
    if missing and args.experiment == 'all':
        print("\n" + "!"*60)
        print("[!] WARNING: Running experiments with incomplete implementations!")
        print("[!] Results will be incomplete or may error.")
        print("!"*60)
    
    setup_directories()
    linear_data, moons_data, digits_data = load_data()
    
    if args.experiment == 'all':
        # Synthetic experiments
        run_synthetic_experiment('linear_gaussian', linear_data, hidden_width=8)
        run_synthetic_experiment('moons', moons_data, hidden_width=32)
        
        # Digits benchmark
        run_digits_experiment(digits_data, track='base')
        
        # Ablations
        run_capacity_ablation(moons_data)
        run_optimizer_study(digits_data)
        run_failure_case_analysis(moons_data)
        
        # Advanced track
        if args.track == 'a':
            run_track_a(digits_data)
        else:
            run_track_b(digits_data)
    
    elif args.experiment == 'linear_gaussian':
        run_synthetic_experiment('linear_gaussian', linear_data)
    
    elif args.experiment == 'moons':
        run_synthetic_experiment('moons', moons_data)
    
    elif args.experiment == 'digits':
        run_digits_experiment(digits_data)
    
    elif args.experiment == 'ablations':
        run_capacity_ablation(moons_data)
        run_optimizer_study(digits_data)
        run_failure_case_analysis(moons_data)
    
    elif args.experiment == 'track_a':
        run_track_a(digits_data)
    
    elif args.experiment == 'track_b':
        run_track_b(digits_data)
    
    elif args.experiment == 'failure':
        run_failure_case_analysis(moons_data)
    
    print("\n" + "="*60)
    print("All experiments completed!")
    print("="*60)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    main()
