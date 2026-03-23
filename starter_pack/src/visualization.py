import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Dict, List, Optional, Tuple
import os

from .models import SoftmaxRegression, OneHiddenLayerNN
from .evaluation import RepeatedSeedResult


def plot_decision_boundary(
    model,
    X: np.ndarray,
    y: np.ndarray,
    resolution: int = 200,
    title: str = "Decision Boundary",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot decision boundary for 2D classification.

    REQUIRED for synthetic tasks (linear gaussian, moons).

    How it works:
        1. Create a grid of points over the feature space
        2. For each grid point, predict class
        3. Color the grid based on predicted class
        4. Overlay actual data points

    Args:
        model: Trained 2D classifier
        X: Feature matrix (n_samples, 2)
        y: Labels (n_samples,)
        resolution: Grid resolution (200 = 200x200 = 40,000 points)
        title: Plot title
        save_path: If provided, save figure
        ax: If provided, plot on this axes

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Define plot bounds with padding
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Create mesh grid
    x_range = np.linspace(x_min, x_max, resolution)
    y_range = np.linspace(y_min, y_max, resolution)
    xx, yy = np.meshgrid(x_range, y_range)
    grid_points = np.c_[xx.ravel(), yy.ravel()]  # (40000, 2)

    # Predict on grid
    Z, _ = model.predict(grid_points)  # labels
    Z = Z.reshape(xx.shape)

    # Plot contours
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)

    # Scatter plot of actual data
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return ax


def plot_training_dynamics(
    history,
    title: str = "Training Dynamics",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot training and validation loss/accuracy over epochs.

    REQUIRED for digits benchmark.

    This plot helps identify:
        - Overfitting (train loss ↓, val loss ↑)
        - Underfitting (both losses high)
        - Good convergence (both losses decrease and stabilize)

    Args:
        history: TrainingHistory object from trainer
        title: Plot title
        save_path: If provided, save figure
        ax: If provided, plot on this axes (for subplots)

    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        return_fig = True
    else:
        fig = ax.figure
        return_fig = False

    epochs = range(1, len(history.train_losses) + 1)

    # Plot both loss and accuracy on the same axes with twin y-axes
    ax2 = ax.twinx()

    # Loss on left y-axis
    line1 = ax.plot(epochs, history.train_losses, 'b-', label='Train Loss', linewidth=2)
    line2 = ax.plot(epochs, history.val_losses, 'b--', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')

    # Accuracy on right y-axis
    line3 = ax2.plot(epochs, history.train_accuracies, 'r-', label='Train Acc', linewidth=2)
    line4 = ax2.plot(epochs, history.val_accuracies, 'r--', label='Val Acc', linewidth=2)
    ax2.set_ylabel('Accuracy', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Combined legend
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right')

    if return_fig and save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_confidence_vs_accuracy(
    confidence_bins: List[Dict],
    title: str = "Confidence vs Accuracy",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confidence bins vs empirical accuracy.

    REQUIRED for Track B: Prediction confidence and reliability.

    This plot shows calibration:
        - If model is well-calibrated: confidence ≈ accuracy
        - If confidence > accuracy: model is overconfident
        - If confidence < accuracy: model is underconfident

    Args:
        confidence_bins: List from Evaluator.confidence_by_bin()
        title: Plot title
        save_path: If provided, save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract data from bins
    bin_centers = [b['mean_confidence'] for b in confidence_bins]
    accuracies = [b['accuracy'] for b in confidence_bins]
    counts = [b['count'] for b in confidence_bins]

    # Create bar chart
    bars = ax.bar(bin_centers, accuracies, width=0.08, alpha=0.7, color='skyblue', edgecolor='black')

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'n={count}', ha='center', va='bottom', fontsize=10)

    # Draw diagonal line (perfect calibration)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')

    # Labels and title
    ax.set_xlabel('Mean Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_optimizer_comparison(
    histories: Dict[str, 'TrainingHistory'],
    title: str = "Optimizer Comparison",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training curves for multiple optimizers.

    REQUIRED for optimizer study ablation.

    Compare: SGD vs Momentum vs Adam

    Expected behavior:
        - SGD: slower convergence, may oscillate
        - Momentum: faster convergence, less oscillation
        - Adam: fast convergence, adaptive learning rates

    Args:
        histories: Dict mapping optimizer name to TrainingHistory
        title: Plot title
        save_path: If provided, save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Define colors for each optimizer
    colors = {'SGD': 'blue', 'Momentum': 'green', 'Adam': 'red'}

    for optimizer_name, history in histories.items():
        epochs = range(1, len(history.val_losses) + 1)
        color = colors.get(optimizer_name, 'black')

        axes[0].plot(epochs, history.val_losses, color=color,
                    label=optimizer_name, linewidth=2)
        axes[1].plot(epochs, history.val_accuracies, color=color,
                    label=optimizer_name, linewidth=2)

    # Loss plot
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Validation Loss')
    axes[0].set_title('Validation Loss Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Accuracy')
    axes[1].set_title('Validation Accuracy Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_capacity_ablation_boundaries(
    models: Dict[int, object],
    X_test: np.ndarray,
    y_test: np.ndarray,
    title: str = "Capacity Ablation: Decision Boundaries",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot decision boundaries for different hidden widths.

    REQUIRED for capacity ablation on moons.

    Shows how increasing capacity changes the boundary:
        - Small width: may underfit (linear-ish boundary)
        - Large width: can capture complex patterns

    Args:
        models: Dict mapping hidden_width to trained model
        X_test: Test features (n_samples, 2)
        y_test: Test labels
        title: Plot title
        save_path: If provided, save figure

    Returns:
        Matplotlib figure with subplots
    """
    n_widths = len(models)
    fig, axes = plt.subplots(1, n_widths, figsize=(6*n_widths, 5))

    if n_widths == 1:
        axes = [axes]  # Make it iterable

    for ax, (width, model) in zip(axes, models.items()):
        plot_decision_boundary(model, X_test, y_test,
                             title=f'Hidden Width = {width}', ax=ax)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_repeated_seed_results(
    results: List[RepeatedSeedResult],
    title: str = "Repeated Seed Evaluation",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot results from repeated seed evaluation with error bars.

    REQUIRED for digits benchmark final reporting.

    Shows:
        - Mean accuracy/loss
        - 95% confidence interval (error bars)
        - Variability across seeds

    Args:
        results: List of RepeatedSeedResult objects
        title: Plot title
        save_path: If provided, save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x_pos = np.arange(len(results))
    labels = [r.model_name for r in results]

    # Accuracy plot with error bars
    means_acc = [r.mean_accuracy for r in results]
    errors_acc = [r.mean_accuracy - r.ci_95_accuracy[0] for r in results]
    axes[0].bar(x_pos, means_acc, yerr=errors_acc, capsize=5,
               color=['skyblue', 'lightgreen'], alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Test Accuracy')
    axes[0].set_title('Mean Test Accuracy ± 95% CI')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(labels)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Loss plot with error bars
    means_loss = [r.mean_loss for r in results]
    errors_loss = [r.mean_loss - r.ci_95_loss[0] for r in results]
    axes[1].bar(x_pos, means_loss, yerr=errors_loss, capsize=5,
               color=['skyblue', 'lightgreen'], alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Test Cross-Entropy Loss')
    axes[1].set_title('Mean Test Loss ± 95% CI')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(labels)
    axes[1].grid(True, alpha=0.3, axis='y')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_pca_scree(
    eigenvalues: np.ndarray,
    title: str = "PCA Scree Plot",
    n_components_show: int = 20,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot PCA scree plot (for Track A).

    Shows:
        - Eigenvalue of each principal component
        - Cumulative variance explained

    Interpretation:
        - Look for "elbow" where curve flattens
        - Components before elbow are most important
        - Cumulative line shows total variance captured

    Args:
        eigenvalues: Eigenvalues from SVD (S² / (n-1))
        title: Plot title
        n_components_show: Number of components to show
        save_path: If provided, save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    n = min(len(eigenvalues), n_components_show)
    components = np.arange(1, n + 1)

    # Bar chart of eigenvalues
    ax.bar(components, eigenvalues[:n], alpha=0.7, color='skyblue', edgecolor='black')

    # Cumulative variance line (twin axis)
    ax2 = ax.twinx()
    cumvar = np.cumsum(eigenvalues[:n]) / np.sum(eigenvalues)
    ax2.plot(components, cumvar, 'ro-', linewidth=2, markersize=6)

    # Labels
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Eigenvalue', color='skyblue')
    ax2.set_ylabel('Cumulative Variance Explained', color='red')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_pca_2d(
    X_pca: np.ndarray,
    y: np.ndarray,
    title: str = "PCA 2D Visualization",
    save_path: Optional[str] = None
) -> plt.Axes:
    """
    Plot 2D PCA visualization of digits data (for Track A).

    Shows:
        - How digits cluster in 2D PCA space
        - Which digits are similar/different

    Args:
        X_pca: PCA-transformed data (n_samples, 2)
        y: Labels (0-9)
        title: Plot title
        save_path: If provided, save figure

    Returns:
        Matplotlib axes
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot with colors by digit
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10',
                        alpha=0.7, edgecolors='black', s=50)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(10))
    cbar.set_label('Digit Label')

    # Labels and title
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return ax


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix.

    Shows:
        - Which classes are confused with each other
        - Diagonal = correct predictions
        - Off-diagonal = misclassifications

    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names
        title: Plot title
        save_path: If provided, save figure

    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Set ticks and labels
    if classes is None:
        classes = [str(i) for i in range(len(cm))]

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")

    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(title)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def save_results_table(
    results: Dict,
    save_path: str
):
    """
    Save experiment results to CSV.

    Args:
        results: Dictionary of results
        save_path: Path to save CSV
    """
    import pandas as pd

    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
