# data_cleaning.py

import pandas as pd

def clean_data(filepath):
    """
    Basic cleaning:
    - Read the data
    - Handle missing values
    - Fix data types
    - Strip column names
    """
    df = pd.read_csv(filepath, delimiter="|", low_memory=False)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Drop rows where both TotalPremium and TotalClaims are missing
    df.dropna(subset=["TotalPremium", "TotalClaims"], how='all', inplace=True)

    # Try to convert date if it exists
    if "TransactionMonth" in df.columns:
        df["TransactionMonth"] = pd.to_datetime(df["TransactionMonth"], errors="coerce")

    # Optional: Fill some important fields
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].fillna("Unknown")

    return df
