"""
PART 1.3: VISUALIZATION & INTERPRETATION (7 points)
====================================================
COMPLETE PLOTTING CODE - NO TODO NEEDED
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)


def plot_outcome_distribution(df, save_path='figures/01_outcome_distribution.png'):
    """
    PLOT 1: Distribution of match outcomes
    Bar chart showing count of Home Win / Draw / Away Win
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    outcome_labels = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
    outcome_counts = df['outcome'].value_counts().sort_index()
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    bars = ax.bar([outcome_labels[i] for i in outcome_counts.index], 
                  outcome_counts.values, 
                  color=colors)
    
    for bar, count in zip(bars, outcome_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Match Outcome', fontsize=12)
    ax.set_ylabel('Number of Matches', fontsize=12)
    ax.set_title('Distribution of Match Outcomes in Dataset', fontsize=14, fontweight='bold')
    ax.set_ylim(0, outcome_counts.max() * 1.15)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plot saved: {save_path}")
    return fig


def plot_goals_distribution(df, save_path='figures/02_goals_distribution.png'):
    """
    PLOT 2: Goals scored distribution (home vs away)
    Overlaid histograms
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(df['home_goals'], bins=range(0, 10), alpha=0.6, 
            label='Home Goals', color='#2ecc71', edgecolor='black')
    ax.hist(df['away_goals'], bins=range(0, 10), alpha=0.6, 
            label='Away Goals', color='#e74c3c', edgecolor='black')
    
    ax.set_xlabel('Number of Goals', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Goals: Home vs Away', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plot saved: {save_path}")
    return fig


def plot_correlation_heatmap(df, numeric_cols, save_path='figures/03_correlation_heatmap.png'):
    """
    PLOT 3: Correlation heatmap of features
    Seaborn heatmap with annotations
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True,
                linewidths=0.5, cbar_kws={'shrink': 0.8})
    
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plot saved: {save_path}")
    return fig


def plot_team_performance(df, save_path='figures/04_team_performance.png'):
    """
    PLOT 4: Top 10 teams by performance
    Bar chart showing goals scored/conceded
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    home_stats = df.groupby('HomeTeam')['home_goals'].sum()
    away_stats = df.groupby('AwayTeam')['away_goals'].sum()
    
    all_teams = set(home_stats.index) | set(away_stats.index)
    team_goals = pd.DataFrame({
        'Scored': [home_stats.get(t, 0) + away_stats.get(t, 0) for t in all_teams]
    }, index=all_teams)
    
    top_teams = team_goals.nlargest(10, 'Scored')
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_teams)))
    bars = ax.barh(range(len(top_teams)), top_teams['Scored'].values, color=colors)
    
    ax.set_yticks(range(len(top_teams)))
    ax.set_yticklabels(top_teams.index)
    ax.set_xlabel('Total Goals Scored', fontsize=12)
    ax.set_title('Top 10 Teams by Goals Scored', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    for bar, val in zip(bars, top_teams['Scored'].values):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                str(int(val)), va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plot saved: {save_path}")
    return fig


def plot_time_series(df, save_path='figures/05_time_series.png'):
    """
    PLOT 5: Match outcomes over time
    Line plot with rolling average
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    df_sorted = df.sort_values('Date')
    
    rolling_home = df_sorted['outcome'].rolling(window=50, min_periods=10).mean()
    
    ax.plot(range(len(df_sorted)), rolling_home, color='#3498db', linewidth=2, label='Rolling Mean (n=50)')
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Draw Line (y=1)')
    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.3, label='Reference')
    ax.axhline(y=1.5, color='red', linestyle='--', alpha=0.3, label='Reference')
    
    ax.set_xlabel('Match Index (Chronological Order)', fontsize=12)
    ax.set_ylabel('Rolling Average Outcome', fontsize=12)
    ax.set_title('Trend of Match Outcomes Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax.set_yticks([0, 0.5, 1, 1.5, 2])
    ax.set_yticklabels(['Home Win (0)', '0.5', 'Draw (1)', '1.5', 'Away Win (2)'])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plot saved: {save_path}")
    return fig


def plot_feature_by_outcome(df, feature='goal_diff', save_path='figures/06_feature_by_outcome.png'):
    """
    PLOT 6: Feature distributions by outcome class
    Violin plots showing feature distribution per outcome
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    outcome_labels = ['Home Win', 'Draw', 'Away Win']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    data_to_plot = [df[df['outcome'] == i][feature] for i in range(3)]
    
    parts = ax.violinplot(data_to_plot, positions=[0, 1, 2], showmeans=True, showmedians=True)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(outcome_labels)
    ax.set_xlabel('Match Outcome', fontsize=12)
    ax.set_ylabel(feature.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Distribution of {feature.replace("_", " ").title()} by Outcome', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plot saved: {save_path}")
    return fig
