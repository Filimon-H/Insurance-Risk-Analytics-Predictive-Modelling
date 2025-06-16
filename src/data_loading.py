import pandas as pd
import numpy as np

def load_data(filepath):
    """Load dataset from a given filepath."""
    data = pd.read_csv(filepath_or_buffer=filepath, delimiter='|', low_memory=False)
    return data

def clean_data(data):
    """Clean the raw data by handling missing values and feature engineering."""

    # Find missing value ratios per column
    column_na_ratios = data.isna().mean()

    # Drop columns with >50% missing values
    to_drop = column_na_ratios[column_na_ratios > 0.5].index
    data = data.drop(columns=to_drop)

    # Drop rows with missing values in columns with <5% missing
    to_drop_rows = column_na_ratios[column_na_ratios < 0.05].index
    data = data.dropna(subset=to_drop_rows)

    # For columns with missing values between 5% and 50%, fill with mode (categorical)
    to_fill = column_na_ratios[(column_na_ratios >= 0.05) & (column_na_ratios <= 0.5)].index

    for col in to_fill:
        if data[col].dtype == object:
            mode_val = data[col].mode()[0]
            data[col] = data[col].fillna(mode_val)
        else:
            mean_val = data[col].mean()
            data[col] = data[col].fillna(mean_val)

    # Feature engineering: create OptimalPremium
    data['OptimalPremium'] = data['TotalPremium'] + np.minimum(0, data['TotalPremium'] - data['TotalClaims']) * -1

    return data
