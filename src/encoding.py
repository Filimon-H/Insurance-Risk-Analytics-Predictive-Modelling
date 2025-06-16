import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_features(data, categorical_features):
    """Encode categorical features using LabelEncoder."""

    new_df = pd.DataFrame()
    label_encoder = LabelEncoder()

    for col in categorical_features:
        new_df[col] = label_encoder.fit_transform(data[col])

    # Add non-categorical features as is
    numerical_features = [col for col in data.columns if col not in categorical_features]
    new_df[numerical_features] = data[numerical_features]

    # Drop any remaining NaNs
    new_df = new_df.dropna()

    return new_df
