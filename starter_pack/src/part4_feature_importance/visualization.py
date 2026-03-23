"""
PART 4.2: INTERPRETATION & VISUALIZATION (7 points)
===================================================
COMPLETE VISUALIZATION CODE - NO TODO NEEDED
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import r2_score, mean_squared_error
import os

os.makedirs('figures', exist_ok=True)


def plot_feature_importance(importance_df, top_n=20, 
                            save_path='figures/feature_importance.png'):
    """Plot top N feature importance."""
    top_df = importance_df.head(top_n).copy()
    
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_df)))
    
    bars = ax.barh(range(len(top_df)), top_df['importance'].values, color=colors)
    
    ax.set_yticks(range(len(top_df)))
    ax.set_yticklabels(top_df['feature'].values)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    for bar, val in zip(bars, top_df['importance'].values):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Feature importance saved: {save_path}")


def plot_shap_summary(shap_values, X_test, feature_names, 
                       save_path='figures/shap_summary.png'):
    """Plot SHAP summary plot."""
    if isinstance(shap_values, list):
        shap_values_plot = np.array(shap_values[0])
    else:
        shap_values_plot = shap_values
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.3)))
    
    shap.summary_plot(shap_values_plot, X_test, feature_names=feature_names,
                      show=False, plot_size=None)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"SHAP summary saved: {save_path}")


def plot_shap_beeswarm(shap_values, X_test, feature_names,
                        save_path='figures/shap_beeswarm.png'):
    """Plot SHAP beeswarm plot showing feature effects."""
    if isinstance(shap_values, list):
        shap_values_plot = np.array(shap_values[0])
    else:
        shap_values_plot = shap_values
    
    plt.figure(figsize=(12, max(6, len(feature_names) * 0.4)))
    shap.summary_plot(shap_values_plot, X_test, feature_names=feature_names,
                      plot_type="dot", show=False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"SHAP beeswarm saved: {save_path}")


def plot_shap_bar(shap_values, feature_names, save_path='figures/shap_bar.png'):
    """Plot SHAP feature importance as bar chart."""
    if isinstance(shap_values, list):
        shap_values_plot = np.array(shap_values[0])
    else:
        shap_values_plot = shap_values
    
    mean_abs_shap = np.mean(np.abs(shap_values_plot), axis=0)
    
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    
    plt.figure(figsize=(10, max(6, len(feature_names) * 0.4)))
    shap.summary_plot(shap_values_plot, X_test, feature_names=feature_names,
                      plot_type="bar", show=False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"SHAP bar chart saved: {save_path}")


def plot_shap_dependence(shap_values, X_test, feature_names, 
                         feature1, feature2=None,
                         save_path='figures/shap_dependence.png'):
    """Plot SHAP dependence plot."""
    if isinstance(shap_values, list):
        shap_values_plot = np.array(shap_values[0])
    else:
        shap_values_plot = shap_values
    
    idx = feature_names.index(feature1)
    
    plt.figure(figsize=(10, 6))
    
    if feature2:
        idx2 = feature_names.index(feature2)
        shap.dependence_plot(feature1, shap_values_plot, X_test,
                            feature_names=feature_names,
                            interaction_index=feature2, show=False)
    else:
        shap.dependence_plot(feature1, shap_values_plot, X_test,
                            feature_names=feature_names, show=False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"SHAP dependence saved: {save_path}")


def plot_shap_force(explainer, shap_values, X_test, feature_names,
                   sample_idx=0, save_path='figures/shap_force.html'):
    """Plot SHAP force plot for single sample."""
    if isinstance(shap_values, list):
        shap_values_sample = shap_values[0][sample_idx]
    else:
        shap_values_sample = shap_values[sample_idx]
    
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values_sample,
        X_test[sample_idx],
        feature_names=feature_names,
        matplotlib=False
    )
    
    shap.save_html(save_path, force_plot)
    print(f"SHAP force plot saved: {save_path}")


def plot_shap_interaction(shap_values, X_test, feature_names,
                          feature1, feature2,
                          save_path='figures/shap_interaction.png'):
    """Plot SHAP interaction values between two features."""
    if isinstance(shap_values, list):
        shap_values_plot = shap_values[0]
    else:
        shap_values_plot = shap_values
    
    idx1 = feature_names.index(feature1)
    idx2 = feature_names.index(feature2)
    
    interaction_values = shap_values_plot[:, idx1] * shap_values_plot[:, idx2]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test[:, idx1], X_test[:, idx2], 
                c=interaction_values, cmap='RdBu_r', alpha=0.6)
    plt.colorbar(label='Interaction Strength')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title(f'SHAP Interaction: {feature1} x {feature2}')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Interaction plot saved: {save_path}")


def plot_comparison_importance(importance_dict, top_n=15,
                               save_path='figures/importance_comparison.png'):
    """
    Compare different importance methods.
    """
    fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.5)))
    
    methods = list(importance_dict.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    
    y_pos = np.arange(top_n)
    
    for i, (method, df) in enumerate(importance_dict.items()):
        top_features = df.head(top_n)['feature'].tolist()
        
        for j, feat in enumerate(top_features):
            ax.barh(y_pos[j] + i * 0.8/len(methods), 1, 
                   height=0.8/len(methods), color=colors[i], 
                   alpha=0.7, label=method if j == 0 else '')
    
    ax.set_yticks(y_pos + 0.4)
    ax.set_yticklabels(top_features)
    ax.set_xlabel('Feature (ranked by each method)')
    ax.set_title('Feature Importance Comparison Across Methods')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Comparison plot saved: {save_path}")


def plot_class_specific_importance(shap_values, X_test, y_test, feature_names,
                                   save_path='figures/class_importance.png'):
    """Plot SHAP importance for each class separately."""
    classes = ['Home Win', 'Draw', 'Away Win']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    
    for i, (ax, class_name) in enumerate(zip(axes, classes)):
        mask = y_test == i
        
        if isinstance(shap_values, list):
            class_shap = np.array(shap_values[i])[mask]
        else:
            class_shap = shap_values[mask]
        
        mean_abs_shap = np.mean(np.abs(class_shap), axis=0)
        
        sorted_idx = np.argsort(mean_abs_shap)[::-1][:10]
        
        ax.barh(range(10), mean_abs_shap[sorted_idx][::-1], color=plt.cm.viridis(0.3 + i * 0.25))
        ax.set_yticks(range(10))
        ax.set_yticklabels([feature_names[j] for j in sorted_idx[::-1]])
        ax.set_xlabel('Mean |SHAP Value|')
        ax.set_title(f'Top Features for {class_name}')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Class-specific importance saved: {save_path}")


def interpret_findings(importance_df, shap_df):
    """
    Generate interpretation report.
    """
    print("=" * 70)
    print("FEATURE IMPORTANCE INTERPRETATION REPORT")
    print("=" * 70)
    
    print("\n📊 TOP 10 MOST IMPORTANT FEATURES:")
    print("-" * 40)
    for i, row in importance_df.head(10).iterrows():
        print(f"  {importance_df.index.get_loc(i)+1}. {row['feature']}: {row['importance']:.4f}")
    
    print("\n📈 KEY INSIGHTS FROM SHAP:")
    print("-" * 40)
    shap_df_sorted = shap_df.sort_values('shap_importance', ascending=False)
    for i, row in shap_df_sorted.head(5).iterrows():
        direction = "increases" if shap_df_sorted.index.get_loc(i) < len(shap_df_sorted) // 2 else "decreases"
        print(f"  • {row['feature']}: Strongly {direction} prediction probability")
    
    print("\n💡 RECOMMENDATIONS:")
    print("-" * 40)
    print("  1. Focus on top features for model improvement")
    print("  2. Consider removing low-importance features to reduce complexity")
    print("  3. Investigate interactions between top features")
    
    return importance_df, shap_df
