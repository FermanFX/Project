"""
PART 1.2: FEATURE ENGINEERING (8 points)
========================================
Mathematical calculations - ALGORITHM parts marked with TODO
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def create_target(df):
    """
    Create target variable (outcome).
    
    ALGORITHM:
    TODO: Implement the formula below
    outcome = 0 (Home Win) if home_goals > away_goals
    outcome = 1 (Draw) if home_goals == away_goals
    outcome = 2 (Away Win) if home_goals < away_goals
    
    HINT: Use np.where() or pd.cut()
    """
    # TODO: Write your implementation here
    pass


def create_goal_diff(df):
    """
    Calculate goal difference feature.
    
    ALGORITHM:
    TODO: Implement the formula below
    goal_diff = home_goals - away_goals
    
    HINT: Simple column subtraction
    """
    # TODO: Write your implementation here
    pass


def create_total_goals(df):
    """
    Calculate total goals per match.
    
    ALGORITHM:
    TODO: Implement the formula below
    total_goals = home_goals + away_goals
    
    HINT: Simple column addition
    """
    # TODO: Write your implementation here
    pass


def create_win_streak_features(df):
    """
    Create rolling win/draw/loss streak features.
    
    ALGORITHM:
    TODO: For each team, calculate:
    - Win streak (consecutive wins)
    - Draw streak (consecutive draws)
    - Loss streak (consecutive losses)
    - Points in last N matches
    
    This requires sorting by date and calculating cumulative stats per team.
    """
    # TODO: Write your implementation here
    pass


def create_elo_rating_features(df):
    """
    Calculate Elo ratings for teams.
    
    ALGORITHM:
    TODO: Implement Elo rating system:
    - Initial rating: 1500
    - K-factor: 32
    - Expected score: E = 1 / (1 + 10^((Ra-Rb)/400))
    - New rating: R_new = R_old + K * (S - E)
    
    Where S = 1 for win, 0.5 for draw, 0 for loss
    """
    # TODO: Write your implementation here
    pass


def encode_categories(df, cat_cols):
    """
    Encode categorical variables.
    
    ALGORITHM:
    TODO: Use LabelEncoder to convert categorical columns to numerical.
    For each categorical column, map unique values to integers [0, 1, 2, ...]
    
    HINT: Use sklearn.preprocessing.LabelEncoder
    """
    # TODO: Write your implementation here
    pass


def select_features(df, target, threshold=0.8):
    """
    Select features based on correlation.
    
    ALGORITHM:
    TODO: 
    1. Calculate correlation matrix for all numeric features
    2. Find pairs with |correlation| > threshold
    3. Remove one feature from each highly correlated pair
    4. Return selected feature names
    """
    # TODO: Write your implementation here
    pass
