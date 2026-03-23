"""
PART 2.3: MODEL EVALUATION & COMPARISON (15 points)
====================================================
COMPLETE EVALUATION CODE - NO TODO NEEDED
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve
)
import seaborn as sns
import os

os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)


def calculate_metrics(y_true, y_pred, y_proba=None, average='macro'):
    """
    Calculate all classification metrics.
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'Recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    if y_proba is not None:
        try:
            metrics['ROC-AUC'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
        except:
            metrics['ROC-AUC'] = None
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names=None, 
                          save_path='figures/confusion_matrix.png', normalize=False):
    """
    Plot confusion matrix heatmap.
    """
    if class_names is None:
        class_names = ['Home Win', 'Draw', 'Away Win']
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Proportion' if normalize else 'Count'})
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''), 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix saved: {save_path}")
    return cm


def plot_roc_curves(y_true, y_proba_dict, save_path='figures/roc_curves.png'):
    """
    Plot ROC curves for all classes.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    class_names = ['Home Win', 'Draw', 'Away Win']
    
    for i, (name, y_proba) in enumerate(y_proba_dict.items()):
        y_true_binary = (np.array(y_true) == i).astype(int)
        
        try:
            fpr, tpr, _ = roc_curve(y_true_binary, y_proba[:, i])
            auc = roc_auc_score(y_true_binary, y_proba[:, i])
            
            ax.plot(fpr, tpr, color=colors[i], linewidth=2,
                   label=f'{name} (AUC = {auc:.3f})')
        except:
            pass
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"ROC curves saved: {save_path}")


def plot_precision_recall_curves(y_true, y_proba_dict, save_path='figures/pr_curves.png'):
    """
    Plot Precision-Recall curves.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for i, (name, y_proba) in enumerate(y_proba_dict.items()):
        y_true_binary = (np.array(y_true) == i).astype(int)
        
        try:
            precision, recall, _ = precision_recall_curve(y_true_binary, y_proba[:, i])
            ax.plot(recall, precision, color=colors[i], linewidth=2, label=name)
        except:
            pass
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"PR curves saved: {save_path}")


def plot_model_comparison(results_dict, save_path='figures/model_comparison.png'):
    """
    Plot bar chart comparing all models.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = list(results_dict.keys())
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    x = np.arange(len(models))
    width = 0.2
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    
    for i, metric in enumerate(metrics_names):
        values = [results_dict[m].get(metric, 0) for m in models]
        ax.bar(x + i * width, values, width, label=metric, color=colors[i])
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Model comparison saved: {save_path}")


def evaluate_all_models(models_dict, X_test, y_test, save_results=True):
    """
    Evaluate all models and generate comprehensive report.
    """
    print("=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    
    all_results = {}
    all_probas = {}
    
    for name, model in models_dict.items():
        print(f"\nEvaluating: {name}")
        
        y_pred = model.predict(X_test)
        
        try:
            y_proba = model.predict_proba(X_test)
            all_probas[name] = y_proba
        except:
            y_proba = None
        
        metrics = calculate_metrics(y_test, y_pred, y_proba)
        all_results[name] = metrics
        
        print(f"  Accuracy:  {metrics['Accuracy']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall:    {metrics['Recall']:.4f}")
        print(f"  F1-Score:  {metrics['F1-Score']:.4f}")
        
        if y_proba is not None:
            print(f"  ROC-AUC:   {metrics.get('ROC-AUC', 'N/A')}")
    
    if save_results:
        results_df = pd.DataFrame(all_results).T
        results_df.to_csv('results/model_results.csv')
        print(f"\nResults saved to: results/model_results.csv")
    
    plot_model_comparison(all_results)
    
    if len(all_probas) > 0:
        plot_roc_curves(y_test, all_probas)
        plot_precision_recall_curves(y_test, all_probas)
    
    return all_results


def cross_validate_model(model, X, y, cv=5):
    """
    Perform cross-validation.
    """
    from sklearn.model_selection import cross_val_score
    
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    print(f"\nCross-Validation Results ({cv}-fold):")
    print(f"  Scores: {scores}")
    print(f"  Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    return scores
