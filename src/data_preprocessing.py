# data_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def handle_missing_data(df):
    df = df.copy()
    df = df.dropna(subset=['TotalClaims', 'CalculatedPremiumPerTerm'])
    df.fillna({'Gender': 'Not specified'}, inplace=True)
    return df


def encode_categorical(df, categorical_cols):
    df = df.copy()
    return pd.get_dummies(df, columns=categorical_cols, drop_first=True)


def split_data(df, target_col, test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def preprocess_claim_severity_data(df, categorical_cols=None, test_size=0.2, random_state=42):
    """
    Preprocess data for claim severity prediction.
    Steps:
    - Filter to only records where 'TotalClaims' > 0 (since severity only applies if there was a claim)
    - Handle missing data
    - Encode categorical variables
    - Split into train and test sets
    
    Parameters:
    - df: Input pandas DataFrame
    - categorical_cols: List of categorical column names to encode (if None, no encoding)
    - test_size: Fraction of data to use as test set
    - random_state: Seed for reproducibility
    
    Returns:
    - X_train, X_test, y_train, y_test
    """
    df = df.copy()
    
    # Filter for records with claims
    df = df[df['TotalClaims'] > 0]
    
    # Handle missing data
    df = handle_missing_data(df)
    
    # Encode categorical variables if specified
    if categorical_cols:
        df = encode_categorical(df, categorical_cols)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df, target_col='TotalClaims', 
                                                  test_size=test_size, 
                                                  random_state=random_state)
    return X_train, X_test, y_train, y_test