"""
PART 3.3: EVALUATION & ANALYSIS (5 points)
===========================================
COMPLETE EVALUATION CODE - NO TODO NEEDED
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import pandas as pd
import os

os.makedirs('figures', exist_ok=True)


def evaluate_nn_model(model, X_test, y_test, class_names=None):
    """
    Comprehensive evaluation of neural network model.
    """
    if class_names is None:
        class_names = ['Home Win', 'Draw', 'Away Win']
    
    model.eval()
    
    X_test_tensor = torch.FloatTensor(X_test)
    
    with torch.no_grad():
        outputs = model(X_test_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predictions = torch.max(outputs, 1)
    
    y_pred = predictions.numpy()
    y_proba = probabilities.numpy()
    
    results = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, average='macro', zero_division=0)
    }
    
    print("=" * 60)
    print("NEURAL NETWORK EVALUATION RESULTS")
    print("=" * 60)
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    return results, y_pred, y_proba


def plot_nn_confusion_matrix(y_true, y_pred, class_names=None, 
                             save_path='figures/nn_confusion_matrix.png'):
    """Plot confusion matrix for neural network."""
    if class_names is None:
        class_names = ['Home Win', 'Draw', 'Away Win']
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Neural Network Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix saved: {save_path}")
    
    return cm


def plot_prediction_confidence(y_proba, y_true, y_pred, 
                               save_path='figures/nn_confidence.png'):
    """Plot prediction confidence analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    max_proba = np.max(y_proba, axis=1)
    
    axes[0].hist(max_proba[y_true == y_pred], bins=20, alpha=0.7, 
                  label='Correct', color='green')
    axes[0].hist(max_proba[y_true != y_pred], bins=20, alpha=0.7, 
                  label='Incorrect', color='red')
    axes[0].set_xlabel('Prediction Confidence (Max Probability)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Prediction Confidence Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    confidence_bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
    accuracy_by_conf = []
    labels = []
    
    for i in range(len(confidence_bins) - 1):
        mask = (max_proba >= confidence_bins[i]) & (max_proba < confidence_bins[i+1])
        if mask.sum() > 0:
            acc = (y_true[mask] == y_pred[mask]).mean()
            accuracy_by_conf.append(acc)
            labels.append(f'{confidence_bins[i]:.1f}-{confidence_bins[i+1]:.1f}')
    
    axes[1].bar(labels, accuracy_by_conf, color='steelblue', alpha=0.8)
    axes[1].set_xlabel('Confidence Range')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy by Confidence Level')
    axes[1].axhline(y=np.mean(y_true == y_pred), color='red', linestyle='--',
                    label='Overall Accuracy')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Confidence analysis saved: {save_path}")


def plot_class_probabilities(y_proba, y_true, class_names=None,
                             save_path='figures/nn_class_probs.png'):
    """Plot probability distribution by true class."""
    if class_names is None:
        class_names = ['Home Win', 'Draw', 'Away Win']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for i, (ax, class_name, color) in enumerate(zip(axes, class_names, colors)):
        mask = y_true == i
        
        for j, (other_class, other_color) in enumerate(zip(class_names, colors)):
            ax.hist(y_proba[mask, j], bins=20, alpha=0.5, 
                   label=f'P({other_class})', color=other_color)
        
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Frequency')
        ax.set_title(f'When True = {class_name}')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Class probabilities saved: {save_path}")


def analyze_nn_weights(model, feature_names=None, 
                       save_path='figures/nn_weight_analysis.png'):
    """Analyze neural network weights."""
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() == 2:
            weights = param.detach().cpu().numpy()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            im = ax.imshow(weights, cmap='RdBu_r', aspect='auto')
            ax.set_xlabel('Output Units')
            ax.set_ylabel('Input Units')
            ax.set_title(f'Weight Matrix: {name}')
            plt.colorbar(im, ax=ax)
            
            plt.tight_layout()
            plt.savefig(f'{save_path[:-4]}_{name.replace(".", "_")}.png', 
                       dpi=150, bbox_inches='tight')
            plt.show()


def save_model(model, path='results/nn_model.pth'):
    """Save trained model."""
    torch.save({
        'model_state_dict': model.state_dict(),
    }, path)
    print(f"Model saved to: {path}")


def load_model(model, path='results/nn_model.pth'):
    """Load trained model."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from: {path}")
    return model
