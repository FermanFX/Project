"""
MAIN SCRIPT - Complete Football Match Prediction Pipeline
Run this script to execute all parts of the project
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.part1_data_analysis import data_processing, feature_engineering, visualization
from src.part2_ml_models import regression_classification, tree_models, evaluation
from src.part3_neural_network import architecture, training, evaluation as nn_evaluation
from src.part4_feature_importance import importance_analysis, visualization as fi_visualization

import pandas as pd
import numpy as np


def main():
    print("=" * 70)
    print("FOOTBALL MATCH OUTCOME PREDICTION - COMPLETE PIPELINE")
    print("=" * 70)
    
    print("\n[1/4] Loading and processing data...")
    df = data_processing.load_data('data/football_data.csv')
    df = data_processing.convert_types(df)
    
    print(f"Dataset shape: {df.shape}")
    
    print("\n[2/4] Feature engineering (TODO sections)...")
    print("NOTE: Complete the TODO sections in src/part1_data_analysis/feature_engineering.py")
    
    print("\n[3/4] Running ML models (TODO sections)...")
    print("NOTE: Complete the TODO sections in src/part2_ml_models/ and src/part3_neural_network/")
    
    print("\n[4/4] Feature importance analysis (TODO sections)...")
    print("NOTE: Complete the TODO sections in src/part4_feature_importance/")
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("\nTo complete the project:")
    print("1. Implement TODO sections in feature_engineering.py")
    print("2. Implement TODO sections in regression_classification.py")
    print("3. Implement TODO sections in tree_models.py")
    print("4. Implement TODO sections in architecture.py")
    print("5. Implement TODO sections in importance_analysis.py")
    print("\nAll plotting code is complete - no TODO needed!")


if __name__ == "__main__":
    main()
