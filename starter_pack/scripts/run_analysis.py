"""
PART 1: DATA ANALYSIS & VISUALIZATION
=====================================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.part1_data_analysis import data_processing, feature_engineering, visualization


def main():
    print("=" * 70)
    print("PART 1: DATA ANALYSIS & VISUALIZATION")
    print("=" * 70)
    
    df = data_processing.load_data('data/football_data.csv')
    
    print("\n1.1 DATA PROCESSING")
    print("-" * 40)
    data_processing.display_info(df)
    data_processing.check_missing(df)
    df = data_processing.check_duplicates(df)
    df = data_processing.convert_types(df)
    
    print("\n1.2 FEATURE ENGINEERING")
    print("-" * 40)
    print("TODO: Complete implementation in src/part1_data_analysis/feature_engineering.py")
    feature_engineering.create_target(df)
    feature_engineering.create_goal_diff(df)
    feature_engineering.create_total_goals(df)
    
    print("\n1.3 VISUALIZATION")
    print("-" * 40)
    print("Generating plots...")
    
    if 'outcome' in df.columns:
        visualization.plot_outcome_distribution(df)
        visualization.plot_goals_distribution(df)
        visualization.plot_team_performance(df)
        visualization.plot_time_series(df)
        visualization.plot_feature_by_outcome(df, 'goal_diff')
    
    print("\n" + "=" * 70)
    print("PART 1 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
