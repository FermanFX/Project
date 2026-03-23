"""
PART 1.1: DATA PROCESSING (10 points)
=====================================
Basic data operations - NO TODO needed here
"""

import pandas as pd
import numpy as np


def load_data(filepath='data/football_data.csv'):
    """Load the football dataset."""
    df = pd.read_csv(filepath)
    return df


def display_info(df):
    """Display dataset shape, columns, dtypes, and first rows."""
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nFirst 5 rows:\n{df.head()}")


def check_missing(df):
    """Check for missing values and return summary."""
    print("\n" + "=" * 60)
    print("MISSING VALUES CHECK")
    print("=" * 60)
    missing = df.isnull().sum()
    print(f"\nMissing values per column:\n{missing[missing > 0]}")
    return missing


def check_duplicates(df):
    """Check and remove duplicate rows."""
    dup_count = df.duplicated().sum()
    print(f"\nDuplicate rows found: {dup_count}")
    df_clean = df.drop_duplicates()
    print(f"Shape after removing duplicates: {df_clean.shape}")
    return df_clean


def convert_types(df):
    """Convert data types if needed."""
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y', errors='coerce')
    return df
